#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::forward {

    __device__ inline float2 compute_ellipse_intersection(
        const float4 conic_opacity,
        const float discriminant,
        const float threshold,
        const float2 mean2d,
        const bool axis_y,
        const float coord)
    {
        const float mean_u = axis_y ? mean2d.y : mean2d.x;
        const float mean_v = axis_y ? mean2d.x : mean2d.y;
        const float coeff = axis_y ? conic_opacity.x : conic_opacity.z;
        const float delta = coord - mean_u;
        const float sqrt_term = sqrtf(discriminant * delta * delta + threshold * coeff);
        return make_float2(
            (-conic_opacity.y * delta - sqrt_term) / coeff + mean_v,
            (-conic_opacity.y * delta + sqrt_term) / coeff + mean_v
        );
    }

    __device__ inline uint process_compact_tiles(
        const float4 conic_opacity,
        const float discriminant,
        const float threshold,
        const float2 mean2d,
        float2 bbox_min,
        float2 bbox_max,
        float2 bbox_argmin,
        float2 bbox_argmax,
        int2 rect_min,
        int2 rect_max,
        const uint grid_width,
        const bool axis_y,
        const uint primitive_idx,
        uint offset,
        const float depth,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices)
    {
        const float block_u = axis_y ? static_cast<float>(config::tile_height) : static_cast<float>(config::tile_width);
        const float block_v = axis_y ? static_cast<float>(config::tile_width) : static_cast<float>(config::tile_height);

        if (axis_y) {
            rect_min = make_int2(rect_min.y, rect_min.x);
            rect_max = make_int2(rect_max.y, rect_max.x);
            bbox_min = make_float2(bbox_min.y, bbox_min.x);
            bbox_max = make_float2(bbox_max.y, bbox_max.x);
            bbox_argmin = make_float2(bbox_argmin.y, bbox_argmin.x);
            bbox_argmax = make_float2(bbox_argmax.y, bbox_argmax.x);
        }

        uint n_touched_tiles = 0;
        float2 intersect_max_line = make_float2(bbox_max.y, bbox_min.y);
        float2 intersect_min_line;
        float min_line = rect_min.x * block_u;

        if (bbox_min.x <= min_line) {
            intersect_min_line = compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, axis_y, min_line);
        }
        else {
            intersect_min_line = intersect_max_line;
        }

        for (int u = rect_min.x; u < rect_max.x; ++u) {
            const float max_line = min_line + block_u;
            if (max_line <= bbox_max.x) {
                intersect_max_line = compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, axis_y, max_line);
            }

            const float ellipse_min = (min_line <= bbox_argmin.y && bbox_argmin.y < max_line)
                ? bbox_min.y
                : fminf(intersect_min_line.x, intersect_max_line.x);
            const float ellipse_max = (min_line <= bbox_argmax.y && bbox_argmax.y < max_line)
                ? bbox_max.y
                : fmaxf(intersect_min_line.y, intersect_max_line.y);

            const int min_tile_v = max(rect_min.y, min(rect_max.y, static_cast<int>(ellipse_min / block_v)));
            const int max_tile_v = min(rect_max.y, max(rect_min.y, static_cast<int>(ellipse_max / block_v + 1.0f)));
            n_touched_tiles += max_tile_v - min_tile_v;

            if (instance_keys != nullptr) {
                for (int v = min_tile_v; v < max_tile_v; ++v) {
                    const uint tile_idx = axis_y
                        ? static_cast<uint>(u * static_cast<int>(grid_width) + v)
                        : static_cast<uint>(v * static_cast<int>(grid_width) + u);
                    const uint depth_key = __float_as_uint(depth);
                    instance_keys[offset] = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key);
                    instance_primitive_indices[offset] = primitive_idx;
                    offset++;
                }
            }

            intersect_min_line = intersect_max_line;
            min_line = max_line;
        }

        return n_touched_tiles;
    }

    __device__ inline uint compact_tile_count_or_emit(
        const float2 mean2d,
        const float4 conic_opacity,
        const uint grid_width,
        const uint grid_height,
        const float compact_box_mult,
        const uint primitive_idx,
        const uint offset,
        const float depth,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices)
    {
        const float discriminant = conic_opacity.y * conic_opacity.y - conic_opacity.x * conic_opacity.z;
        if (conic_opacity.x <= 0.0f || conic_opacity.z <= 0.0f || discriminant >= 0.0f) return 0;

        const float threshold = compact_box_mult * 2.0f * logf(conic_opacity.w * config::min_alpha_threshold_rcp);
        if (threshold <= 0.0f) return 0;

        float x_term = sqrtf(-(conic_opacity.y * conic_opacity.y * threshold) / (discriminant * conic_opacity.x));
        x_term = (conic_opacity.y < 0.0f) ? x_term : -x_term;
        float y_term = sqrtf(-(conic_opacity.y * conic_opacity.y * threshold) / (discriminant * conic_opacity.z));
        y_term = (conic_opacity.y < 0.0f) ? y_term : -y_term;

        const float2 bbox_argmin = make_float2(mean2d.y - y_term, mean2d.x - x_term);
        const float2 bbox_argmax = make_float2(mean2d.y + y_term, mean2d.x + x_term);
        const float2 bbox_min = make_float2(
            compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, true, bbox_argmin.x).x,
            compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, false, bbox_argmin.y).x
        );
        const float2 bbox_max = make_float2(
            compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, true, bbox_argmax.x).y,
            compute_ellipse_intersection(conic_opacity, discriminant, threshold, mean2d, false, bbox_argmax.y).y
        );

        const int2 rect_min = make_int2(
            max(0, min(static_cast<int>(grid_width), static_cast<int>(bbox_min.x / static_cast<float>(config::tile_width)))),
            max(0, min(static_cast<int>(grid_height), static_cast<int>(bbox_min.y / static_cast<float>(config::tile_height))))
        );
        const int2 rect_max = make_int2(
            max(0, min(static_cast<int>(grid_width), static_cast<int>(bbox_max.x / static_cast<float>(config::tile_width) + 1.0f))),
            max(0, min(static_cast<int>(grid_height), static_cast<int>(bbox_max.y / static_cast<float>(config::tile_height) + 1.0f)))
        );

        const int x_span = rect_max.x - rect_min.x;
        const int y_span = rect_max.y - rect_min.y;
        if (x_span * y_span == 0) return 0;

        return process_compact_tiles(
            conic_opacity,
            discriminant,
            threshold,
            mean2d,
            bbox_min,
            bbox_max,
            bbox_argmin,
            bbox_argmax,
            rect_min,
            rect_max,
            grid_width,
            y_span < x_span,
            primitive_idx,
            offset,
            depth,
            instance_keys,
            instance_primitive_indices
        );
    }

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        float* __restrict__ primitive_depth,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases_rest,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float compact_box_mult)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // z culling
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) return;

        // load opacity
        const float opacity = opacities[primitive_idx];
        if (opacity < config::min_alpha_threshold) return;

        // compute 3d covariance from scale and rotation
        const float3 scale = scales[primitive_idx];
        const float3 variance = scale * scale;
        const float4 rotation = rotations[primitive_idx];
        const mat3x3 R = convert_normalized_quaternion_to_rotation_matrix(rotation);
        const mat3x3 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        const mat3x3_triu cov3d {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };

        // compute 2d mean in normalized image coordinates
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // ewa splatting
        const float clip_left = (-0.15f * width - center_x) / focal_x;
        const float clip_right = (1.15f * width - center_x) / focal_x;
        const float clip_top = (-0.15f * height - center_y) / focal_y;
        const float clip_bottom = (1.15f * height - center_y) / focal_y;
        const float x_clipped = clamp(x, clip_left, clip_right);
        const float y_clipped = clamp(y, clip_top, clip_bottom);
        const float j11 = focal_x / depth;
        const float j13 = -j11 * x_clipped;
        const float j22 = focal_y / depth;
        const float j23 = -j22 * y_clipped;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z
        );
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z
        );
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
        );
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2)
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < config::min_cov2d_determinant) return; // or (determinant <= 0.0f) with explicit handling in backward
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant
        );

        // 2d mean in screen space
        const float2 mean2d = make_float2(
            x * focal_x + center_x,
            y * focal_y + center_y
        );

        const float4 conic_opacity = make_float4(conic, opacity);
        const uint n_touched_tiles = compact_tile_count_or_emit(
            mean2d,
            conic_opacity,
            grid_width,
            grid_height,
            compact_box_mult,
            primitive_idx,
            0,
            depth,
            nullptr,
            nullptr
        );
        if (n_touched_tiles == 0) return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(0, 0, 0, 0);
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = conic_opacity;
        primitive_color[primitive_idx] = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest, mean3d, cam_position[0], primitive_idx,
            active_sh_bases, total_sh_bases_rest
        );
        primitive_depth[primitive_idx] = depth;
    }

    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float* __restrict__ primitive_depths,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint grid_height,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float compact_box_mult,
        const uint n_primitives)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        compact_tile_count_or_emit(
            primitive_mean2d[primitive_idx],
            primitive_conic_opacity[primitive_idx],
            grid_width,
            grid_height,
            compact_box_mult,
            primitive_idx,
            offset,
            primitive_depths[primitive_idx],
            instance_keys,
            instance_primitive_indices
        );
    }

    __global__ void extract_instance_ranges_cu(
        const uint64_t* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const uint64_t instance_key = instance_keys[instance_idx];
        const uint instance_tile_idx = instance_key >> 32;
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint64_t previous_instance_key = instance_keys[instance_idx - 1];
            const uint previous_instance_tile_idx = previous_instance_key >> 32;
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void count_tile_buckets_cu(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_bucket_counts,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint2 range = tile_instance_ranges[tile_idx];
        const uint n_instances = range.y - range.x;
        tile_bucket_counts[tile_idx] = div_round_up(n_instances, 32u);
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const uint* __restrict__ tile_bucket_offsets,
        uint* __restrict__ bucket_to_tile,
        float* __restrict__ sampled_transmittance,
        float* __restrict__ sampled_accumulated_color,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ tile_final_transmittances,
        uint* __restrict__ tile_n_contrib,
        uint* __restrict__ tile_max_contrib,
        float* __restrict__ tile_pixel_colors,
        const int* __restrict__ metric_map,
        int* __restrict__ metric_counts,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ uint collected_primitive_idx[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        __shared__ uint max_contrib_scratch[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint contributor = 0;
        uint last_contributor = 0;
        bool done = !inside;
        // collaborative loading and processing
        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const uint n_tile_instances = tile_range.y - tile_range.x;
        const uint n_tile_buckets = div_round_up(n_tile_instances, 32u);
        for (uint bucket_idx = thread_rank; bucket_idx < n_tile_buckets; bucket_idx += config::block_size_blend) {
            bucket_to_tile[bucket_offset + bucket_idx] = tile_idx;
        }
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_idx[thread_rank] = primitive_idx;
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_color[thread_rank] = color;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                if ((j & 31) == 0) {
                    const uint bucket_pixel_offset = bucket_offset * config::block_size_blend + thread_rank;
                    const uint bucket_color_offset = bucket_offset * config::block_size_blend * 3 + thread_rank;
                    sampled_transmittance[bucket_pixel_offset] = transmittance;
                    sampled_accumulated_color[bucket_color_offset] = color_pixel.x;
                    sampled_accumulated_color[config::block_size_blend + bucket_color_offset] = color_pixel.y;
                    sampled_accumulated_color[2 * config::block_size_blend + bucket_color_offset] = color_pixel.z;
                    bucket_offset++;
                }

                contributor++;

                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                if (!config::original_stability_measures) exponent = fminf(exponent, 0.0f);
                else if (exponent > 0.0f) continue;
                const float gaussian = expf(exponent);
                const float fragment_alpha = opacity * gaussian;
                if (fragment_alpha < config::min_alpha_threshold) continue;
                const float alpha = config::original_stability_measures ? fminf(fragment_alpha, config::max_fragment_alpha) : fragment_alpha;

                // compute remaining transmittance after this fragment
                const float next_transmittance = transmittance * (1.0f - alpha);

                // early stopping as in original 3DGS, i.e., before blending (if config::original_stability_measures)
                if (config::original_stability_measures && next_transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }

                // blend fragment into pixel color
                color_pixel += transmittance * alpha * collected_color[j];

                if (metric_counts != nullptr) {
                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                    if (metric_map[pixel_idx] != 0) {
                        atomicAdd(&metric_counts[collected_primitive_idx[j]], 1);
                    }
                }

                // update transmittance
                transmittance = next_transmittance;
                last_contributor = contributor;

                // early stopping (if not config::original_stability_measures)
                if (!config::original_stability_measures && transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
        if (inside) {
            // apply background color
            color_pixel += transmittance * bg_color[0];
            // store results
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint n_pixels = width * height;
            tile_pixel_colors[pixel_idx] = color_pixel.x;
            tile_pixel_colors[n_pixels + pixel_idx] = color_pixel.y;
            tile_pixel_colors[2 * n_pixels + pixel_idx] = color_pixel.z;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            tile_final_transmittances[pixel_idx] = transmittance;
            tile_n_contrib[pixel_idx] = last_contributor;
        }

        max_contrib_scratch[thread_rank] = last_contributor;
        block.sync();
        for (uint stride = config::block_size_blend / 2; stride > 0; stride >>= 1) {
            if (thread_rank < stride) {
                const uint other = max_contrib_scratch[thread_rank + stride];
                if (other > max_contrib_scratch[thread_rank]) max_contrib_scratch[thread_rank] = other;
            }
            block.sync();
        }
        if (thread_rank == 0) {
            tile_max_contrib[tile_idx] = max_contrib_scratch[0];
        }
    }

}
