#pragma once

#include "rasterization_config.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace fast_gs::rasterization::kernels::metric_counts {

    // Count how many high-error pixels each Gaussian contributed to.
    // metric_map: (H*W,) int32 — 1 = high-error pixel, 0 = normal pixel.
    // counts:     (N,)   float32 — accumulated in-place (caller zeros before calling).
    __global__ void __launch_bounds__(config::block_size_blend) compute_metric_counts_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const int* __restrict__ metric_map,
        float* __restrict__ counts,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index  = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank  = block.thread_rank();

        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width  + thread_index.x,
            group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;

        const int metric_val = inside
            ? metric_map[pixel_coords.y * width + pixel_coords.x]
            : 0;

        __shared__ uint   collected_primitive_idx[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];

        const float2 pixel = make_float2(
            __uint2float_rn(pixel_coords.x),
            __uint2float_rn(pixel_coords.y)) + 0.5f;

        float transmittance = 1.0f;
        bool done = !inside;

        const uint2 tile_range =
            tile_instance_ranges[group_index.y * grid_width + group_index.x];

        for (int n_points_remaining = static_cast<int>(tile_range.y - tile_range.x),
                 current_fetch_idx = static_cast<int>(tile_range.x) + static_cast<int>(thread_rank);
             n_points_remaining > 0;
             n_points_remaining -= config::block_size_blend,
             current_fetch_idx  += config::block_size_blend)
        {
            if (__syncthreads_count(done) == config::block_size_blend) break;

            if (current_fetch_idx < static_cast<int>(tile_range.y)) {
                const uint primitive_idx             = instance_primitive_indices[current_fetch_idx];
                collected_primitive_idx[thread_rank] = primitive_idx;
                collected_mean2d[thread_rank]        = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
            }
            block.sync();

            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float4 co      = collected_conic_opacity[j];
                const float3 conic   = make_float3(co);
                const float  opacity = co.w;
                const float2 delta   = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x
                                              + conic.z * delta.y * delta.y)
                                             - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                if (!config::original_opacity_interpretation
                    && gaussian < config::min_alpha_threshold) continue;
                const float alpha = opacity * gaussian;
                if (config::original_opacity_interpretation
                    && alpha < config::min_alpha_threshold) continue;

                // Pixel contributes to this Gaussian: if it is a high-error pixel, count it.
                if (metric_val == 1) {
                    atomicAdd(&counts[collected_primitive_idx[j]], 1.0f);
                }

                transmittance *= 1.0f - alpha;
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                }
            }
        }
    }

}
