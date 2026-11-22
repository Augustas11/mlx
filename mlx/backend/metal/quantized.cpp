// Copyright © 2023-2026 Apple Inc.

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/small_m_route.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename... Args>
auto get_quantized_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_quantized_kernel(d, name, template_def, mode);
}

template <typename... Args>
auto get_qmm_nax_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_qmm_nax_kernel(d, name, template_def, mode);
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen >= 17 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) {
      return 33;
    } else if (D <= 4096 && O <= 4096) {
      return 25;
    } else {
      return 13;
    }
  } else if (arch_gen >= 15 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) {
      return 13;
    } else if (D <= 4096 && O <= 4096) {
      return 15;
    } else {
      return 13;
    }
  } else if (arch_gen >= 13) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return (arch_gen == 13 && arch_size == 's') ? 7 : 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

// Must match the K step in qmv_fast_impl (kernels/quantized.h):
//   pack_factor<bits, 32>() * (bits == 2 ? 1 : 2) * SIMD_SIZE
inline int qmv_fast_k_alignment(int bits) {
  return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    int offset) {
  if (skip) {
    return offset;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }

  return offset;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(strides[0], offset++);
  compute_encoder.set_vector_bytes(strides[1], offset++);

  return offset;
}

auto get_quantize_kernel_dims(
    MTL::ComputePipelineState* kernel,
    const array& w,
    const array& out,
    int group_size,
    int bits,
    bool dequantize = false) {
  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;
  int per_thread =
      dequantize ? packs_per_int : std::max(group_size / simd_size, 1);
  size_t nthreads =
      dequantize ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  return std::make_tuple(grid_dims, group_dims);
}

void quantize_impl(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    QuantizationMode mode,
    int group_size,
    int bits,
    bool dequantize,
    Stream s) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  bool has_biases = (mode == QuantizationMode::Affine);
  bool has_global_scale = !has_biases && (inputs.size() > (1 + dequantize));

  auto w = ensure_row_contiguous(w_pre, d, s);
  if (dequantize) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    if (has_biases) {
      auto biases = ensure_row_contiguous(inputs[2], d, s);
      compute_encoder.set_input_array(biases, 2);
    } else if (has_global_scale) {
      compute_encoder.set_input_array(inputs[2], 2);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    scales.set_data(allocator::malloc(scales.nbytes()));
    if (has_biases) {
      auto& biases = outputs[2];
      biases.set_data(allocator::malloc(biases.nbytes()));
      compute_encoder.set_output_array(biases, 3);
    } else if (has_global_scale) {
      compute_encoder.set_input_array(inputs[1], 3);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
  }

  auto type_string = dequantize ? get_type_string(out.dtype())
                                : get_type_string(w_pre.dtype());
  auto mode_string = quantization_mode_to_string(mode);
  std::string kname;
  concatenate(
      kname,
      mode_string + (dequantize ? "_dequantize" : "_quantize"),
      "_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  if (!has_biases) {
    concatenate(kname, "_hgs_", has_global_scale ? "true" : "false");
  }
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      dequantize ? "dequantize" : "quantize",
      mode_string,
      type_string,
      group_size,
      bits,
      has_global_scale);

  auto [grid_dims, group_dims] =
      get_quantize_kernel_dims(kernel, w, out, group_size, bits, dequantize);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

auto quantize_input(
    const array& w,
    const std::optional<array>& global_scale,
    QuantizationMode mode,
    int group_size,
    int bits,
    metal::Device& d,
    Stream s) {
  auto wq_shape = w.shape();
  wq_shape.back() = w.shape(-1) * bits / 32;
  auto scales_shape = w.shape();
  scales_shape.back() = w.shape(-1) / group_size;

  std::vector<array> inputs{w};
  if (global_scale) {
    inputs.push_back(*global_scale);
  }
  std::vector<array> outputs{
      array(wq_shape, uint32, nullptr, {}),
      array(scales_shape, uint8, nullptr, {})};
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(outputs[0]);
  compute_encoder.add_temporary(outputs[1]);
  quantize_impl(inputs, outputs, mode, group_size, bits, false, s);
  return std::make_tuple(outputs[0], outputs[1]);
}

void fp_quantize_dequantize(
    const array& in,
    const std::optional<array>& global_scale,
    array& out,
    const std::string& mode,
    int group_size,
    int bits,
    metal::Device& d,
    const Stream& s) {
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(in, d, s);
  compute_encoder.set_input_array(w, 0);
  if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 1);
  }
  compute_encoder.set_output_array(out, 2);
  auto type_string = get_type_string(in.dtype());
  std::string kname;
  concatenate(
      kname,
      mode + "_quantize_dequantize_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_hgs_",
      global_scale ? "true" : "false");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      "quantize_dequantize",
      mode,
      type_string,
      group_size,
      bits,
      global_scale.has_value());

  auto [grid_dims, group_dims] =
      get_quantize_kernel_dims(kernel, w, out, group_size, bits);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

array quantize_dequantize_input(
    const array& x_pre,
    const std::optional<array>& global_scale,
    const std::string& mode,
    int group_size,
    int bits,
    metal::Device& d,
    Stream s) {
  bool donate_x = x_pre.is_donatable();
  array x = ensure_row_contiguous(x_pre, d, s);
  // If x is a copy it should be donatable
  donate_x |= x.is_donatable();
  auto xhat =
      donate_x ? x : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
  if (!donate_x) {
    metal::get_command_encoder(s).add_temporary(xhat);
  }
  fp_quantize_dequantize(x, global_scale, xhat, mode, group_size, bits, d, s);
  return xhat;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());

  concatenate(
      kname,
      mode + "_qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_quad", mode, type_string, group_size, bits, K, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = (K % qmv_fast_k_alignment(bits) == 0);
  bool partial_rows = fast && (N % bn != 0);
  // A narrower output tile reduces register pressure for large
  // floating-point quantized matrix-vector products on M5 Max GPUs.
  bool use_narrow_qmv = fast && N >= 4096 && d.get_architecture_gen() == 17 &&
      d.get_architecture().back() == 's' && mode == "nvfp4";
  int results_per_simdgroup = use_narrow_qmv ? 2 : 4;
  bn = 2 * results_per_simdgroup;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  concatenate(
      kname,
      mode + (fast ? "_qmv_fast_" : "_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0",
      partial_rows ? "_pr_1" : "_pr_0",
      global_scale ? "_hgs_1" : "_hgs_0",
      "_r_",
      results_per_simdgroup);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "qmv_fast" : "qmv"),
      mode,
      type_string,
      group_size,
      bits,
      B > 1,
      partial_rows,
      global_scale.has_value(),
      results_per_simdgroup);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// affine qmv_wide only beats qmv on gen-15+; fp benefits on every gen.
inline bool use_qmv_wide(const std::string& mode, metal::Device& d) {
  return mode != "affine" || d.get_architecture_gen() >= 15;
}

// Dispatches qmv_wide (fp modes -> fp_qmv_wide, affine -> affine_qmv_wide):
// vecs_per_tg input vectors streamed and reused per weight group.
void qmv_wide(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // vecs_per_tg is the per-threadgroup input-vector tile. Each tile re-reads
  // the weights, so use the fewest tiles, then the smallest tile that fills
  // them.
  int n_tiles = (M + 4) / 5; // ceil(M / 5); tile size caps at 5
  int vecs_per_tg = (M + n_tiles - 1) / n_tiles;

  // k_lanes: lanes reducing K per output row (32/k_lanes rows per simdgroup).
  // The affine subchunk decode has enough ALU per weight load to favor more
  // rows per simdgroup (kl8); the fp modes' vectorized dot is balanced at 16.
  int k_lanes = mode == "affine" ? 8 : 16;
  constexpr int num_simdgroups = 2;
  int B = out.size() / M / N;
  bool batched = B > 1;
  // Output rows per threadgroup: (32 / k_lanes) per simdgroup x num_simdgroups.
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;

  MTL::Size group_dims(32, num_simdgroups, 1);
  MTL::Size grid_dims(
      (M + vecs_per_tg - 1) / vecs_per_tg,
      (N + rows_per_tg - 1) / rows_per_tg,
      B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qmv_wide_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_nv_",
      vecs_per_tg,
      "_kl_",
      k_lanes,
      batched ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      "qmv_wide",
      mode,
      type_string,
      group_size,
      bits,
      vecs_per_tg,
      k_lanes,
      batched);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, !batched, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// macprovider: MLX_METAL_SMALL_M_ROW_INVARIANT selects the small-M route of
// the weights use_affine_qmv_mma accepts:
//   unset or 1 (or any value other than 0 and 2): the row-invariant
//     qmv_mma1 / qmv_mma pair for 1..16 rows;
//   0: the upstream routes (qmv / qmm);
//   2: "stock order": 1 row takes the upstream route; exactly 2 rows run the
//     upstream one-row kernel's per-row arithmetic (affine_qmv_fast_rows
//     where one row takes qmv_fast, affine_qmv with the row in the grid
//     where one row takes plain qmv); 3..16 rows take qmv_mma exactly as
//     the default route does; more rows are unchanged (see
//     QuantizedMatmul::eval_gpu). Guarantees: a row of a 2-row call equals
//     that row computed alone, bit for bit; the rows of 3..16-row calls
//     equal each other across those row counts; a row of a 3+ row call can
//     differ from the lone row in low bits (qmv_mma sums K in another
//     order).
//     Why: measured on an M3 Ultra (generation 15), every quantized linear
//     of a real 4-bit model chained once per pass, ms per pass at rows
//     1 / 2 / 3 / 4 / 5 / 8 / 16:
//       Llama 3.1 8B  qmv_fast_rows 2..16: 8.76 / 12.27 / 16.22 / 20.24 /
//                       23.05 / 32.92 / 59.53
//                     route 1 (WK 8):     9.96 / 12.85 / 12.89 / 12.93 /
//                       12.96 / 13.01 / 22.08
//                     route 0:            8.75 / 10.62 / 13.69 / 16.96 /
//                       20.50 / 29.35 / 44.88
//       27B model     qmv_fast_rows 2..16: 26.45 / 36.51 / 48.79 / 61.42 /
//                       71.77 / 105.66 / 209.64
//                     route 1:            30.38 / 39.05 / 39.08 / 39.13 /
//                       38.34 / 39.34 / 79.35
//                     route 0:            26.55 / 32.04 / 43.23 / 54.31 /
//                       65.63 / 95.15 / 154.32
//     One row on the upstream route matches route 0; two rows on
//     qmv_fast_rows are 4-7% faster than qmv_mma (12.27 vs 12.85, 36.51 vs
//     39.05); from three rows qmv_mma is faster. The 2-row plain-qmv
//     shapes (K not a multiple of the qmv_fast K step) were not measured.
// The process value is read once (process_small_m_route in small_m_route.cpp):
// a route that could change inside a process would let state computed on one
// route continue on the other. A thread may scope ops it builds to route 1 or
// 2 (ScopedSmallMRoute); each QuantizedMatmul captures the route in effect
// when it is added to the graph (small_m_route_), and this file reads only
// that member, so an op keeps its route whatever scope is active when it is
// evaluated.

// macprovider: selects the affine weights whose small-M rows take the qmv_mma
// reduction (ml-explore/mlx#4787) on non-NAX GPUs, ahead of the qmm route (see
// QuantizedMatmul::eval_gpu). Every test here reads the weight shape, type and
// device, never the row count M, so a row takes the same reduction order alone
// and inside a batch. Upstream starts qmv_mma at 3 or 4 rows and keeps Ultra
// on qmv; both checks depend on M or the chip size, so they are dropped and
// the window starts at 1 row, where qmv_mma1 computes the same bits.
// The stock-order route (route 2) uses the same weight test.
inline bool use_affine_qmv_mma(
    int route,
    const array& w,
    const array& scales,
    QuantizationMode mode,
    int group_size,
    int bits,
    int N,
    int K) {
  if (mode != QuantizationMode::Affine || metal::is_nax_available()) {
    return false;
  }
  // macprovider: the row-invariant window exists so a row decodes to the same
  // bits alone and inside a batch. A process that never puts two decode rows
  // in one product for its whole lifetime does not need it and can start with
  // MLX_METAL_SMALL_M_ROW_INVARIANT=0 to take the per-row qmv / qmm route
  // below, which is what upstream runs and is faster at one row. The default
  // keeps the window, so leaving the variable unset can cost speed, never a
  // row's bits. Route 0 is process-wide only (a scope cannot select it).
  if (route == small_m_route_serial_only) {
    return false;
  }
  if ((bits != 4 && bits != 8) ||
      (group_size != 32 && group_size != 64 && group_size != 128)) {
    return false;
  }
  if ((w.offset() % 16) || (scales.offset() % 2)) {
    return false;
  }
  return (K % 128 == 0) && (N >= 512) && (K >= 512) &&
      (int64_t(N) * K >= (int64_t(1) << 21)) && (N % 16 == 0);
}

// macprovider: WK, the number of K partial sums of the qmv_mma reduction, for
// both qmv_mma and qmv_mma1 (see affine_qmv_mma1 in kernels/quantized.h).
// The default is 8 on the stock-order route (route 2) on
// generation 13 and 15 GPUs, and 4 everywhere else: the row-invariant route
// keeps 4 on every GPU, so its bits are the ones native MTP and the other
// row-invariant processes were qualified on. MLX_METAL_QMV_MMA_WK = 4, 8 or
// 16 overrides the default (any other value keeps it). Measured on an M3 Ultra (generation 15), every quantized linear
// of a 4-bit model chained once per pass, ms per pass at 1 / 2 / 4 / 8 / 16
// rows: Llama 3.1 8B 10.45 / 13.69 / 13.80 / 13.93 / 23.01 at WK 4 against
// 9.68 / 12.91 / 12.93 / 13.03 / 22.08 at WK 8; a 27B model 31.93 / 40.76 /
// 40.75 / 40.98 / 81.70 against 30.42 / 39.05 / 39.18 / 39.46 / 79.45.
// WK 16 was slower than 8 at one row on the 27B model. On an M1 (generation
// 13, 8 GB) the same benchmark on a 3B model gave 34.73 / 40.45 / 52.19 /
// 61.58 at 1 / 2 / 4 / 8 rows with WK 4 against 32.60 / 41.18 / 42.06 /
// 53.64 with WK 8 (one run per cell). Other generations that take this
// route (14, 16, and non-NAX 17) were not measured and keep 4. A row alone and the
// same row in a batch gave the same bits at 4, 8 and 16 on that GPU (the
// small-M tests). WK changes the reduction order and so the bits of every
// output, so it is a function of the route alone within a process (the GPU
// generation and the variable are read once). quantized_matmul captures it on
// the QuantizedMatmul next to the route when the op is built
// (qmv_mma_wk_), and both kernels take that value. The kernel names carry
// WK (qmv_mma always, qmv_mma1 when WK != 4), so the pipelines of both
// routes coexist in one process.
int metal::qmv_mma_partial_sums(int route) {
  static const int gen =
      metal::device(mlx::core::Device::gpu).get_architecture_gen();
  static const int env_wk = env::get_var("MLX_METAL_QMV_MMA_WK", 0);
  const int fallback =
      (route == small_m_route_stock_order && (gen == 13 || gen == 15)) ? 8
                                                                         : 4;
  return (env_wk == 4 || env_wk == 8 || env_wk == 16) ? env_wk : fallback;
}

void qmv_mma(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode,
    int wk) {
  // wk sets a row's K reduction order and must match qmv_mma1; the row
  // tiling never changes it, tm and bn only pick how many rows and columns
  // one threadgroup covers. The threadgroup is 32 wk threads (at most 512).
  int tm = (M > 8) ? 2 : 1;
  int bn = 16;
  // Below gen 15, affine with one row tile is faster at 32 columns (M2 Pro).
  if ((d.get_architecture_gen() < 15) && (tm == 1) && (N % 32 == 0)) {
    bn = 32;
  }

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode,
      "_qmv_mma_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_wk_",
      wk,
      "_bn_",
      bn,
      "_tm_",
      tm);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_mma", mode, type_string, group_size, bits, wk, bn, tm);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  int rows_per_tg = 8 * tm;
  compute_encoder.dispatch_threadgroups(
      MTL::Size(N / bn, (M + rows_per_tg - 1) / rows_per_tg, 1),
      MTL::Size(32, wk, 1));
}

// macprovider: rows up to this count take qmv_mma1, more rows take qmv_mma.
// This is a speed choice only: the two kernels give the same bits for a row
// (see affine_qmv_mma1 in kernels/quantized.h), so moving the switch point
// changes no result. qmv_mma1 pays for every row and qmv_mma shares one read
// of the weights across its rows. Measured on an M3 Ultra with Llama 3.1 8B
// 4-bit, aggregate tok/s: 1 row 93 on qmv_mma1 against 75 on qmv_mma; 2 rows
// 120 on qmv_mma1 against 146 on qmv_mma.
constexpr int qmv_mma1_max_rows = 1;

// One threadgroup per 32 / wk output columns of one row; the grid height is
// the row count. Any M works since a row never reads another row.
void qmv_mma1(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode,
    int wk) {
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname, mode, "_qmv_mma1_", type_string, "_gs_", group_size, "_b_", bits);
  // wk = 4 keeps the kernel name it had before wk was a parameter.
  if (wk != 4) {
    concatenate(kname, "_wk_", wk);
  }
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_mma1", mode, type_string, group_size, bits, wk);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.dispatch_threadgroups(
      MTL::Size(N / (32 / wk), M, 1), MTL::Size(32, 1, 1));
}

// macprovider: the 2-row stock-order kernel (MLX_METAL_SMALL_M_ROW_INVARIANT=2,
// see the route comment above use_affine_qmv_mma). affine_qmv_fast_rows
// (kernels/quantized.h) computes rows in one dispatch with, for every
// output element, exactly the float
// operations of upstream's affine_qmv_fast for that row alone, so a row
// equals that row computed alone by qmv_fast bit for bit. A standalone harness
// found 0 differing elements against per-row qmv_fast on an Apple M5 and an
// Apple M3 Ultra, and the small-M tests passed on the M3 Ultra with it
// serving 2..16 rows. The route sends it 2 rows only: in the model passes
// measured in that comment it beat qmv_mma at 2 rows and lost from 3.
// MR = 4 rows per threadgroup; 4 bits keep the raw weight words in registers
// (packed), 8 bits keep unpacked floats, the faster variant of each in the M3
// Ultra timing. Grid (ceil(M / 4), N / 8, 1), threadgroup (32, 2, 1), as in
// the harness. Callers guarantee K % qmv_fast_k_alignment(bits) == 0,
// N % 8 == 0, bits 4 or 8, a row-contiguous [M, K] x and no batch.
void qmv_fast_rows(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  constexpr int mr = 4;
  const bool packed = (bits == 4);
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode,
      "_qmv_fast_rows_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_mr_",
      mr,
      "_p_",
      packed ? 1 : 0);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      "qmv_fast_rows",
      mode,
      type_string,
      group_size,
      bits,
      mr,
      packed);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  compute_encoder.set_input_array(*biases, c++);
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.dispatch_threadgroups(
      MTL::Size((M + mr - 1) / mr, N / 8, 1), MTL::Size(32, 2, 1));
}

void qvm_split_k(
    array x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  auto& compute_encoder = metal::get_command_encoder(s);

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;

  // When weights are 2D, collapse leading batch dimensions of x and out into M
  // so the kernel and reduction operate on standard 2D matrices.
  if (B == 1 && x.ndim() != 2) {
    x = reshape_in_eval(x, {M, K}, s);
  }
  if (B == 1 && out.ndim() != 2) {
    out = reshape_in_eval(out, {M, N}, s);
  }

  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);
  MTL::Size grid_dims = MTL::Size(M, (N + bn - 1) / bn, B);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm_split_k", mode, type_string, group_size, bits, split_k);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (biases) {
    auto b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0",
      global_scale ? "_hgs" : "");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      "qvm",
      mode,
      type_string,
      group_size,
      bits,
      B > 1,
      global_scale.has_value());
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  // Use smaller bm when one block covers all of M. Only qmm_t_nax has a 32-row
  // instantiation.
  int bm = (transpose && M <= 32) ? 32 : 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_nax_" : "_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0",
      global_scale ? "_hgs" : "");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn,
        global_scale.has_value());
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  } else if (transpose) {
    if (global_scale) {
      compute_encoder.set_input_array(*global_scale, c);
    }
    c++;
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  // The gather qmm NAX kernels are instantiated with BK = 64 only; any
  // other value here makes the kernel name lookup fail.
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      global_scale ? "_hgs" : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        bm,
        bk,
        bn,
        wm,
        wn,
        global_scale.has_value());
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        bm,
        bk,
        bn,
        wm,
        wn);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  assert(
      (!global_scale || (transpose && !biases)) &&
      "qmm global scale requires transposed weights without biases");

  bool has_nax_kernel =
      metal::is_nax_available() && (transpose || mode == "affine");
  bool nax_aligned = (K % 64 == 0) && (transpose || N % 64 == 0);
  if (has_nax_kernel && nax_aligned &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* const std::optional<array>& global_scale = */ global_scale,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_" : "_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0",
      global_scale ? "_hgs" : "");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        global_scale.has_value());
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_n", mode, type_string, group_size, bits, batched);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  } else if (transpose) {
    if (global_scale) {
      compute_encoder.set_input_array(*global_scale, c);
    }
    c++;
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int split_m,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // Choose split_k to target ~512 threadgroups
  int bm = 32, bn = 32;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int split_m_tiles = (split_m + bm - 1) / bm;
  int current_tgs = n_tiles * split_m_tiles;
  int split_k = std::max(1, 512 / current_tgs);

  // Each K partition must be a whole number of BK-wide (32) K-tiles as well as
  // whole quantization groups. The qmm_t_splitk kernels tile K by BK=32 and do
  // not bound the K dimension, so a partition smaller than BK (e.g. nvfp4's
  // group_size=16) would over-read into the next group's weights/scales.
  int k_align = group_size > 32 ? group_size : 32;
  split_k = std::min(split_k, K / k_align);

  // Ensure K divides evenly by split_k * k_align
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(
        x,
        w,
        scales,
        biases,
        std::nullopt,
        out,
        true,
        group_size,
        bits,
        M,
        N,
        K,
        d,
        s,
        mode);
  }

  int k_partition_size = K / split_k;
  int split_k_partition_stride = M * N;

  // Allocate a float32 buffer for the partial sums, so that each partition is
  // rounded to the output type only once, after they're summed, as in
  // steel_matmul_splitk.
  auto& compute_encoder = metal::get_command_encoder(s);
  array intermediate({split_k, M, N}, float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  // Grid: (N_tiles, M_tiles, split_k)
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);

  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Sum the partitions and cast to the output type
  auto accum_kname = "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
      type_to_name(intermediate);
  auto accum_kernel = get_steel_gemm_splitk_accum_kernel(
      d, accum_kname, intermediate, out, false);
  compute_encoder.set_compute_pipeline_state(accum_kernel);
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(split_k, 2);
  compute_encoder.set_bytes(split_k_partition_stride, 3);
  compute_encoder.set_bytes(N, 4);
  compute_encoder.dispatch_threads(MTL::Size(N, M, 1), get_block_dims(N, M, 1));
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return gather_qmm_nax(
        /* const array& x = */ x,
        /* const array& w = */ w,
        /* const array& scales = */ scales,
        /* const std::optional<array>& biases = */ biases,
        /* const std::optional<array>& global_scale = */ global_scale,
        /* const array& lhs_indices = */ lhs_indices,
        /* const array& rhs_indices = */ rhs_indices,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string& mode = */ mode);
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      global_scale ? "_hgs" : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        global_scale.has_value());
  } else {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n",
        mode,
        type_string,
        group_size,
        bits,
        global_scale.has_value());
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;
  concatenate(
      kname,
      mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      global_scale ? "_hgs" : "");

  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "gather_qmv_fast" : "gather_qmv"),
      mode,
      type_string,
      group_size,
      bits,
      global_scale.has_value());

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_gather_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      global_scale ? "_hgs" : "");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      "gather_qvm",
      mode,
      type_string,
      group_size,
      bits,
      global_scale.has_value());
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_input_array(scales, 1);
  if (biases) {
    compute_encoder.set_input_array(*biases, 2);
  } else if (global_scale) {
    compute_encoder.set_input_array(*global_scale, 2);
  }
  int c = 3;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c++);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs_nax(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const std::optional<array>& global_scale,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);
  std::optional<array> biases;
  if (biases_) {
    biases = ensure_row_contiguous(*biases_, d, s);
  }
  std::optional<array> gs;
  if (global_scale) {
    gs = ensure_row_contiguous(*global_scale, d, s);
  }

  // Use smaller bm for many experts and few tokens.
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int bm = (M / E < 64) ? 32 : 64;
  int bn = 64, bk = 64;
  int wm = 2, wn = 2;

  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode +
          (transpose ? "_gather_qmm_rhs_nax_nt_" : "_gather_qmm_rhs_nax_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn,
      global_scale ? "_hgs" : "");

  metal::MTLFCList func_consts = {
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  array offsets = gather_mm_offsets(indices, E, M, d, s);

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_nax_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose,
      global_scale.has_value());
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims(
      (N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(scales, 2);
  if (biases) {
    compute_encoder.set_input_array(*biases, 3);
  } else if (gs) {
    compute_encoder.set_input_array(*gs, 3);
  }
  int c = 4;
  compute_encoder.set_input_array(offsets, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(E, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const std::optional<array>& global_scale,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  if (metal::is_nax_available() &&
      (env::enable_tf32() || x_.dtype() != float32)) {
    return gather_qmm_rhs_nax(
        /* const array& x_ = */ x_,
        /* const array& w_ = */ w_,
        /* const array& scales_ = */ scales_,
        /* const std::optional<array>& biases_ = */ biases_,
        /* const std::optional<array>& global_scale = */ global_scale,
        /* const array& indices_ = */ indices_,
        /* array& out = */ out,
        /* bool transpose = */ transpose,
        /* int group_size = */ group_size,
        /* int bits = */ bits,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* metal::Device& d = */ d,
        /* const Stream& s = */ s,
        /* const std::string mode = */ mode);
  }

  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of x broadcasted
  // with rhs_indices. We need only broadcast x and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);
  std::optional<array> biases;
  if (biases_) {
    biases = ensure_row_contiguous(*biases_, d, s);
  }
  std::optional<array> gs;
  if (global_scale) {
    gs = ensure_row_contiguous(*global_scale, d, s);
  }

  // TODO: Tune the block sizes
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;

  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn,
      global_scale ? "_hgs" : "");

  metal::MTLFCList func_consts = {
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  array offsets = gather_mm_offsets(indices, E, M, d, s);

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose,
      global_scale.has_value());
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims(
      (N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(scales, 2);
  if (biases) {
    compute_encoder.set_input_array(*biases, 3);
  } else if (gs) {
    compute_encoder.set_input_array(*gs, 3);
  }
  int c = 4;
  compute_encoder.set_input_array(offsets, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(E, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode,
    bool reference_prefill = false) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((K == 128 || (K == 64 && bits >= 2)) && is_power_of_2(bits) &&
      !global_scale) {
    qmv_quad(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }

  // Flagged multi-token prefill follows the vendor route. Decode and ordinary
  // primitives keep the batch-invariant qmv path below.
  if (reference_prefill && M >= 2 && use_qmv_wide(mode, d) && !global_scale) {
    qmv_wide(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }

  // macprovider: upstream sends M = 1 to qmv and 2 <= M < vector_limit to
  // qmv_wide. The two reduce a vector in different orders, so a row's bits
  // would follow how many rows share the call. Here every affine row count
  // below the qmm route takes qmv_wide, M = 1 included, for the shapes where
  // qmv_wide beats qmv. qmv_wide sums each vector in an order set by k_lanes
  // and the weight group alone; the vector tile (vecs_per_tg) only decides
  // how many vectors share one read of the weights, so a row matches its
  // serial result while the weights are read once per tile of up to 5 rows
  // instead of once per row.
  // The route reads the weight shape and the device, never M:
  // - K not a multiple of 128 (gpt-oss, K = 2880) only. Aligned-K shapes
  //   that miss qmv_mma (N < 512) stay on qmv, where qmv_wide cost GLM-5-Next
  //   0.7% of 4-row decode.
  // - Gen 15 and 16 only. On gen 17 (M5) qmv_wide lost about a fifth of
  //   single-row Qwen3-8B decode speed to qmv.
  int gen = d.get_architecture_gen();
  if (mode == "affine" && gen >= 15 && gen <= 16 && (K % 128) != 0 &&
      biases && !global_scale) {
    qmv_wide(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }

  qmv(x,
      w,
      scales,
      biases,
      global_scale,
      out,
      group_size,
      bits,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);
  bool reference_prefill =
      reference_prefill_ && non_batched && transpose_ && x.ndim() >= 3 &&
      x.shape(-2) > 1;
  int route_m = reference_prefill ? x.shape(-2) : M;

  // macprovider: rows below the qmm route take the qmv_mma reduction when the
  // weight shape qualifies, from 1 row up: WK partial sums over 128-wide K
  // blocks, added in one fixed order (WK per route, see qmv_mma_partial_sums;
  // the op carries its value in qmv_mma_wk_). That is the one reduction order
  // of the op's route, and two kernels implement it: qmv_mma1 for 1 row and
  // qmv_mma (ml-explore/mlx#4787) for 2 rows and more. Both give the
  // same bits for a row, so a row's result does not depend on how many rows
  // share the call, and the switch point is a speed choice (qmv_mma1_max_rows). The
  // weight test reads no row count, and the row limit is at least 17 so 1..16
  // rows always match bit for bit, where upstream would send rows from
  // vector_limit up to qmm. An unaligned or non-contiguous x is copied into
  // one [rows, K] block rather than sent to qmv, so the route cannot follow
  // how a caller sliced its rows.
  // Bit equality of the two kernels and of every row count is checked by
  // test_qmm_small_m_mma1_bit_equal and test_qmm_small_m_batch_invariant
  // (python/tests/test_quantized.py) and by "test gpu small-M quantized matmul
  // batch invariance" (tests/gpu_tests.cpp).
  if (transpose_ && !reference_prefill && w.ndim() == 2 && biases &&
      use_affine_qmv_mma(
          small_m_route_, w, scales, mode_, group_size_, bits_, N, K)) {
    int rows = x.size() / K;
    const bool stock_order = (small_m_route_ == small_m_route_stock_order);
    // quantized_matmul sets qmv_mma_wk_ for every affine GPU op; an op built
    // for another device and moved here takes its route's value.
    const int wk = (qmv_mma_wk_ == 4 || qmv_mma_wk_ == 8 || qmv_mma_wk_ == 16)
        ? qmv_mma_wk_
        : metal::qmv_mma_partial_sums(small_m_route_);
    if (stock_order && rows == 2) {
      // macprovider: stock order, 2 rows (see the route comment above
      // use_affine_qmv_mma for the measurement). Each row must equal that
      // row computed alone on the upstream one-row route, bit for bit.
      // Under the weight test above
      // (affine, no NAX, bits 4 or 8, K % 128 == 0, K >= 512, N % 16 == 0,
      // biases, 2-D w) one row (M = 1, below every vector_limit) never
      // reaches qmm, qmv_quad (K is 64 or 128 only), qmv_wide (K % 128 != 0
      // only) or a NAX path: it takes qmv(), which runs affine_qmv_fast
      // (no partial rows, N % 8 == 0) when K % qmv_fast_k_alignment(bits)
      // == 0 (512 at 4 bits, 256 at 8) and plain affine_qmv otherwise (e.g.
      // K = 1152, or K = 768 at 4 bits).
      // - qmv_fast shapes: qmv_fast_rows, which runs qmv_fast's per-row float
      //   operations with each K block's weights read once for both rows.
      // - other shapes: the same qmv() call with M = 2, i.e. the one-row
      //   kernel with the row in the grid (grid (M, N / 8, 1)). affine_qmv
      //   (qmv_impl in kernels/quantized.h) takes its row from tid.x only:
      //   it offsets x and y by tid.x * K and tid.x * N and never reads M,
      //   so each threadgroup runs exactly the instructions of the M = 1
      //   call for that row. The pipeline is the same instantiation too:
      //   B = 1 (batch_0), partial_rows false, no global scale, 4 results
      //   per simdgroup. One dispatch was chosen over two one-row
      //   dispatches because it is the same per-row code with one launch.
      // Not qmv_wide, upstream MLX's choice for 2 rows below vector_limit:
      // it reduces a row in a different order than the one-row kernel.
      // Calling qmv() here keeps the 2-row result independent of
      // vector_limit and of dispatch_qmv's routing.
      if (!x.flags().row_contiguous) {
        x = contiguous_copy_gpu(x, s);
        metal::get_command_encoder(s).add_temporary(x);
      }
      if (K % qmv_fast_k_alignment(bits_) == 0) {
        qmv_fast_rows(
            x,
            w,
            scales,
            biases,
            out,
            group_size_,
            bits_,
            rows,
            N,
            K,
            d,
            s,
            mode);
      } else {
        qmv(x,
            w,
            scales,
            biases,
            std::nullopt,
            out,
            group_size_,
            bits_,
            rows,
            N,
            K,
            d,
            s,
            mode);
      }
      return;
    }
    // macprovider: stock order sends 3..16 rows here, to the same qmv_mma
    // call as the default route (rows > qmv_mma1_max_rows); 1 row and more
    // than 16 rows fall through to the upstream route below.
    if (stock_order ? (rows >= 3 && rows <= 16)
                    : (rows > 0 && rows < std::max(vector_limit, 17))) {
      if (!x.flags().row_contiguous || (x.offset() % (8 * x.itemsize()))) {
        x = contiguous_copy_gpu(x, s);
        metal::get_command_encoder(s).add_temporary(x);
      }
      if (rows <= qmv_mma1_max_rows) {
        qmv_mma1(
            x,
            w,
            scales,
            biases,
            out,
            group_size_,
            bits_,
            rows,
            N,
            K,
            d,
            s,
            mode,
            wk);
      } else {
        qmv_mma(
            x,
            w,
            scales,
            biases,
            out,
            group_size_,
            bits_,
            rows,
            N,
            K,
            d,
            s,
            mode,
            wk);
      }
      return;
    }
  }

  // It is a matrix matrix product.
  if (route_m >= vector_limit) {
    // LFM2.5 opts into the vendor's split-K accumulation for multi-token
    // prefill. Pick the split count from one row's sequence length so a
    // shared B-row prefill reduces K exactly like B independent rows.
    if (reference_prefill) {
      qmm_splitk(
          x,
          w,
          scales,
          biases,
          out,
          group_size_,
          bits_,
          M,
          x.shape(-2),
          N,
          K,
          d,
          s,
          mode);
      return;
    }
    // macprovider: no qmm_splitk. Its split count follows the M tile count,
    // so the same row sums K in a different order when the call carries more
    // rows (a 2-row prefill chunk vs the 1-row serial chunk). qmm reduces K
    // in one fixed order for every M, which keeps rows batch-invariant.
    qmm(x,
        w,
        scales,
        biases,
        std::nullopt,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    dispatch_qmv(
        x,
        w,
        scales,
        biases,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode,
        reference_prefill);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    qvm(x,
        w,
        scales,
        biases,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(
      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  // Affine gets biases at index 3, nvfp4 an optional global scale.
  std::optional<array> biases = std::nullopt;
  std::optional<array> global_scale = std::nullopt;
  if (mode_ == QuantizationMode::Affine) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  } else if (inputs.size() == 6) {
    global_scale = ensure_row_contiguous(inputs[3], d, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);

  // We are walking x in order and w is also in order so we can batch up the
  // matmuls and reuse reading x and w.
  //
  // TODO: Tune 16 and 4 here a bit better.
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        global_scale,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        global_scale,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        global_scale,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      global_scale,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& x_pre = inputs[0];
  const array& w_pre = inputs[1];
  auto mode = quantization_mode_to_string(mode_);

  out.set_data(allocator::malloc(out.nbytes()));

  // - 2 inputs: x, w (non-quantized w)
  // - 3 inputs: x, w, scales_w (quantized w)
  bool w_quantized = (inputs[1].dtype() == uint32);
  int base_size = w_quantized ? 3 : 2;
  // For nvfp4, global scales are optional but must be both present or both
  // absent If present, they add 2 more inputs (global_scale_x, global_scale_w)
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
  assert(inputs.size() == base_size || has_global_scales);

  std::optional<array> global_scale_x;
  std::optional<array> global_scale_w;
  if (has_global_scales) {
    global_scale_x = inputs[inputs.size() - 2];
    global_scale_w = inputs[inputs.size() - 1];
  }

  // Quantize weights.
  auto [w_q, scales_w] = !w_quantized
      ? quantize_input(w_pre, global_scale_w, mode_, group_size_, bits_, d, s)
      : std::make_tuple(
            ensure_row_contiguous_matrix(w_pre, d, s),
            ensure_row_contiguous_matrix(inputs[base_size - 1], d, s));

  // Quantize activation.
  array x = quantize_dequantize_input(
      x_pre, global_scale_x, mode, group_size_, bits_, d, s);

  bool non_batched = w_q.ndim() == 2;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);
  if (has_global_scales && w_quantized && non_batched &&
      x.dtype() == bfloat16 && K % 32 == 0 &&
      M >= get_qmv_batch_limit(K, N, d)) {
    qmm(x,
        w_q,
        scales_w,
        std::nullopt,
        global_scale_w,
        out,
        true,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }
  dispatch_qmv(
      x,
      w_q,
      scales_w,
      std::nullopt,
      global_scale_w,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode,
      false);
}

void GatherQQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& x_pre = inputs[0];
  const array& w_pre = inputs[1];
  const array& lhs_indices = ensure_row_contiguous(inputs[2], d, s);
  const array& rhs_indices = ensure_row_contiguous(inputs[3], d, s);
  auto mode = quantization_mode_to_string(mode_);

  out.set_data(allocator::malloc(out.nbytes()));

  // - 4 inputs: x, w (non-quantized w)
  // - 5 inputs: x, w, scales_w (quantized w)
  bool w_quantized = (inputs[1].dtype() == uint32);
  int base_size = w_quantized ? 5 : 4;
  // For nvfp4, global scales are optional but must be both present or both
  // absent If present, they add 2 more inputs (global_scale_x, global_scale_w)
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
  assert(inputs.size() == base_size || has_global_scales);

  std::optional<array> global_scale_x;
  std::optional<array> global_scale_w;
  if (has_global_scales) {
    global_scale_x = inputs[inputs.size() - 2];
    global_scale_w = inputs[inputs.size() - 1];
  }

  // Quantize weights.
  auto [w_q, scales_w] = !w_quantized
      ? quantize_input(w_pre, global_scale_w, mode_, group_size_, bits_, d, s)
      : std::make_tuple(
            ensure_row_contiguous_matrix(w_pre, d, s),
            ensure_row_contiguous_matrix(inputs[base_size - 1], d, s));

  // Quantize activation.
  array x = quantize_dequantize_input(
      x_pre, global_scale_x, mode, group_size_, bits_, d, s);

  bool non_batched = w_q.ndim() == 2;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  // temporary, until we add proper scaling for gather qqmm
  if (global_scale_w) {
    int E = w_q.size() / w_q.shape(-1) / w_q.shape(-2);
    array gs_e(Shape{E}, float32, nullptr, {});
    broadcast(*global_scale_w, gs_e);
    global_scale_w = ensure_row_contiguous(gs_e, d, s);
  }

  int B = out.size() / M / N;
  bool use_matrix_kernels = has_global_scales && w_quantized &&
      x.dtype() == bfloat16 && w_q.ndim() == 3 &&
      K % (metal::is_nax_available() ? 64 : 32) == 0;

  if (use_matrix_kernels && M == 1 && B >= 16 && right_sorted_) {
    int E = w_q.size() / w_q.shape(-1) / w_q.shape(-2);
    if (B / E >= 4) {
      gather_qmm_rhs(
          x,
          w_q,
          scales_w,
          std::nullopt,
          global_scale_w,
          rhs_indices,
          out,
          true,
          group_size_,
          bits_,
          x.size() / K,
          N,
          K,
          d,
          s,
          mode);
      return;
    }
  }

  if (use_matrix_kernels && M >= get_qmv_batch_limit(K, N, d)) {
    gather_qmm(
        x,
        w_q,
        scales_w,
        std::nullopt,
        global_scale_w,
        lhs_indices,
        rhs_indices,
        out,
        true,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qmv(
      x,
      w_q,
      scales_w,
      std::nullopt,
      global_scale_w,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  quantize_impl(
      inputs, outputs, mode_, group_size_, bits_, dequantize_, stream());
}

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  unary_op_gpu(inputs, out, name(), stream());
}

} // namespace mlx::core
