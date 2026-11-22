// Copyright © 2026 Apple Inc.

#include <sstream>
#include <stdexcept>

#include "mlx/small_m_route.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int no_scope = -1;

int& thread_scope() {
  thread_local int scope = no_scope;
  return scope;
}

void check_scope_route(int route) {
  if (route != small_m_route_row_invariant &&
      route != small_m_route_stock_order) {
    std::ostringstream msg;
    msg << "[small_m_route] A route scope selects 1 (row_invariant) or 2 "
        << "(stock_order), got " << route << ".";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

// Read once: a route that could change inside a process would let state
// computed on one route continue on the other. Per-op routes come from
// scopes instead, which each op captures when it is built.
int process_small_m_route() {
  static const int route = [] {
    int v = env::get_var(
        "MLX_METAL_SMALL_M_ROW_INVARIANT", small_m_route_row_invariant);
    return (v == small_m_route_serial_only || v == small_m_route_stock_order)
        ? v
        : small_m_route_row_invariant;
  }();
  return route;
}

int small_m_route() {
  int scope = thread_scope();
  return scope == no_scope ? process_small_m_route() : scope;
}

void set_small_m_route_scope(int route) {
  check_scope_route(route);
  thread_scope() = route;
}

void clear_small_m_route_scope() {
  thread_scope() = no_scope;
}

int small_m_route_scope() {
  return thread_scope();
}

ScopedSmallMRoute::ScopedSmallMRoute(int route) : previous_(thread_scope()) {
  set_small_m_route_scope(route);
}

ScopedSmallMRoute::~ScopedSmallMRoute() {
  thread_scope() = previous_;
}

} // namespace mlx::core
