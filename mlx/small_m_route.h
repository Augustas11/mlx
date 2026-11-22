// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/api.h"

namespace mlx::core {

// macprovider: the small-row route of affine quantized matmul on non-NAX
// GPUs (see small_m_route in backend/metal/quantized.cpp for what each
// route computes):
//   0 serial_only    upstream per-row kernels, process-wide only;
//   1 row_invariant  the qmv_mma window, a row equals itself in any batch;
//   2 stock_order    a 2-row product equals each lone row, 3..16 rows
//                    equal each other.
// The process route is read once from MLX_METAL_SMALL_M_ROW_INVARIANT with
// C atoi semantics: 0 and 2 select those routes, anything else (or unset)
// selects 1. A thread may override it with a scope (ScopedSmallMRoute) for
// the ops it builds; a scope selects 1 or 2, never 0. The route is captured
// by each QuantizedMatmul when the op is added to the graph, and it joins
// the compile cache key, so the scope only needs to be set while the graph
// is built (or a compiled function is called), not while it is evaluated.
constexpr int small_m_route_serial_only = 0;
constexpr int small_m_route_row_invariant = 1;
constexpr int small_m_route_stock_order = 2;

/** The process route, read once from MLX_METAL_SMALL_M_ROW_INVARIANT. */
MLX_API int process_small_m_route();

/** The route ops built on this thread capture: the scope, else the process
 * route. */
MLX_API int small_m_route();

/** Set this thread's route scope (1 or 2; anything else throws). Prefer
 * ScopedSmallMRoute; this entry exists for C bindings. */
MLX_API void set_small_m_route_scope(int route);

/** Clear this thread's route scope so ops take the process route again. */
MLX_API void clear_small_m_route_scope();

/** This thread's scope, or -1 when none is set. */
MLX_API int small_m_route_scope();

/** Sets this thread's route scope for its lifetime and restores the previous
 * scope (or none) on destruction. */
class MLX_API ScopedSmallMRoute {
 public:
  explicit ScopedSmallMRoute(int route);
  ~ScopedSmallMRoute();
  ScopedSmallMRoute(const ScopedSmallMRoute&) = delete;
  ScopedSmallMRoute& operator=(const ScopedSmallMRoute&) = delete;

 private:
  int previous_;
};

} // namespace mlx::core
