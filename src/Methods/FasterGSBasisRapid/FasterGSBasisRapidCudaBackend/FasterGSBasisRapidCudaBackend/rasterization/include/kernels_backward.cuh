#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::backward {

    struct GradientSums {
        float color_x;
        float color_y;
        float color_z;
        float opacity;
        float conic_x;
        float conic_y;
        float conic_z;
        float mean_x;
        float mean_y;
        float mean_abs_x;
        float mean_abs_y;
    };

    struct GradientSumsAdd {
        __device__ __forceinline__ GradientSums operator()(const GradientSums& a, const GradientSums& b) const {
            return {
                a.color_x + b.color_x,
                a.color_y + b.color_y,
                a.color_z + b.color_z,
                a.opacity + b.opacity,
                a.conic_x + b.conic_x,
                a.conic_y + b.conic_y,
                a.conic_z + b.conic_z,
                a.mean_x + b.mean_x,
                a.mean_y + b.mean_y,
                a.mean_abs_x + b.mean_abs_x,
                a.mean_abs_y + b.mean_abs_y,
            };
        }
    };

    __global__ void preprocess_backward_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float2* __restrict__ grad_mean2d_abs,
        const float* __restrict__ grad_conic,
        const float* __restrict__ grad_colors,
        float3* __restrict__ grad_means,
        float3* __restrict__ grad_scales,
        float4* __restrict__ grad_rotations,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
        float* __restrict__ densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_sh_bases_rest,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_0, sh_coefficients_rest, grad_colors, grad_sh_coefficients_0, grad_sh_coefficients_rest,
            mean3d, cam_position[0], primitive_idx, n_primitives, active_sh_bases, total_sh_bases_rest
        );

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

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

        // ewa splatting gradient helpers
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

        // 2d covariance gradient
        const float a = dot(jwc_r1, jw_r1) + config::dilation, b = dot(jwc_r1, jw_r2), c = dot(jwc_r2, jw_r2) + config::dilation;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float determinant_sq = determinant * determinant;
        const float determinant_rcp_sq = 1.0f / determinant_sq; // denominator options: (determinant_sq + eps), fmaxf(determinant_sq, eps)
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]
        );
        const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
            2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
            bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
            2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z
        );

        // 3d covariance gradient
        const mat3x3_triu dL_dcov3d = {
            jw_r1.x * jw_r1.x * dL_dcov2d.x + 2.0f * jw_r1.x * jw_r2.x * dL_dcov2d.y + jw_r2.x * jw_r2.x * dL_dcov2d.z,
            jw_r1.x * jw_r1.y * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.y * dL_dcov2d.z,
            jw_r1.x * jw_r1.z * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.z * dL_dcov2d.z,
            jw_r1.y * jw_r1.y * dL_dcov2d.x + 2.0f * jw_r1.y * jw_r2.y * dL_dcov2d.y + jw_r2.y * jw_r2.y * dL_dcov2d.z,
            jw_r1.y * jw_r1.z * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + jw_r2.y * jw_r2.z * dL_dcov2d.z,
            jw_r1.z * jw_r1.z * dL_dcov2d.x + 2.0f * jw_r1.z * jw_r2.z * dL_dcov2d.y + jw_r2.z * jw_r2.z * dL_dcov2d.z,
        };

        // gradient of J * W
        const float3 dL_djw_r1 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
            jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
            jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y
        );
        const float3 dL_djw_r2 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
            jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
            jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z
        );

        // gradient of non-zero entries in J
        const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
        const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
        const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
        const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

        // load gradient of 2d mean
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];

        // for adaptive density control
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            const float2 dL_dmean2d_ndc = 0.5f * make_float2(
                dL_dmean2d.x * width,
                dL_dmean2d.y * height
            );
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d_ndc);
            const float2 dL_dmean2d_abs = grad_mean2d_abs[primitive_idx];
            const float2 dL_dmean2d_abs_ndc = 0.5f * make_float2(
                dL_dmean2d_abs.x * width,
                dL_dmean2d_abs.y * height
            );
            densification_info[2 * n_primitives + primitive_idx] += length(dL_dmean2d_abs_ndc);
        }

        // mean3d camera space gradient from mean2d
        float3 dL_dmean3d_cam = make_float3(
            j11 * dL_dmean2d.x,
            j22 * dL_dmean2d.y,
            -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y
        );

        // add mean3d camera space gradient from J while accounting for clipping
        const bool valid_x = x >= clip_left && x <= clip_right;
        const bool valid_y = y >= clip_top && y <= clip_bottom;
        if (valid_x) dL_dmean3d_cam.x -= j11 * dL_dj13 / depth;
        if (valid_y) dL_dmean3d_cam.y -= j22 * dL_dj23 / depth;
        const float factor_x = 1.0f + static_cast<float>(valid_x);
        const float factor_y = 1.0f + static_cast<float>(valid_y);
        dL_dmean3d_cam.z += (j11 * (factor_x * x_clipped * dL_dj13 - dL_dj11) + j22 * (factor_y * y_clipped * dL_dj23 - dL_dj22)) / depth;

        // 3d mean gradient from splatting
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z
        );

        // write total 3d mean gradient
        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        grad_means[primitive_idx] = dL_dmean3d;

        // scale gradient
        const float3 dL_dvariance = make_float3(
            R.m11 * R.m11 * dL_dcov3d.m11 + R.m21 * R.m21 * dL_dcov3d.m22 + R.m31 * R.m31 * dL_dcov3d.m33 +
                2.0f * (R.m11 * R.m21 * dL_dcov3d.m12 + R.m11 * R.m31 * dL_dcov3d.m13 + R.m21 * R.m31 * dL_dcov3d.m23),
            R.m12 * R.m12 * dL_dcov3d.m11 + R.m22 * R.m22 * dL_dcov3d.m22 + R.m32 * R.m32 * dL_dcov3d.m33 +
                2.0f * (R.m12 * R.m22 * dL_dcov3d.m12 + R.m12 * R.m32 * dL_dcov3d.m13 + R.m22 * R.m32 * dL_dcov3d.m23),
            R.m13 * R.m13 * dL_dcov3d.m11 + R.m23 * R.m23 * dL_dcov3d.m22 + R.m33 * R.m33 * dL_dcov3d.m33 +
                2.0f * (R.m13 * R.m23 * dL_dcov3d.m12 + R.m13 * R.m33 * dL_dcov3d.m13 + R.m23 * R.m33 * dL_dcov3d.m23)
        );
        const float3 dL_dscale = 2.0f * dL_dvariance;
        grad_scales[primitive_idx] = dL_dscale;

        // rotation gradient
        const mat3x3 dL_dR = {
            2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13),
            2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13),
            2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13),
            2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23),
            2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23),
            2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23),
            2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33),
            2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33),
            2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33)
        };
        const float4 dL_drotation = convert_normalized_quaternion_to_rotation_matrix_backward(rotation, dL_dR);
        grad_rotations[primitive_idx] = dL_drotation;

    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ image,
        const float* __restrict__ tile_final_transmittances,
        float2* __restrict__ grad_mean2d,
        float2* __restrict__ grad_mean2d_abs,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_opacity,
        float* __restrict__ grad_colors,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const uint thread_rank = block.thread_rank();
        const uint local_x = thread_rank % config::tile_width;
        const uint local_y = thread_rank / config::tile_width;
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + local_x, group_index.y * config::tile_height + local_y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ uint collected_primitive_idx[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        using GradientBlockReduce = cub::BlockReduce<GradientSums, config::block_size_blend>;
        __shared__ typename GradientBlockReduce::TempStorage gradient_reduce_storage;
        // initialize local storage
        const float3 background = bg_color[0];
        float3 color_pixel_residual, grad_color_pixel;
        float grad_alpha_common;
        if (inside) {
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint n_pixels = width * height;
            // final values from forward pass before background blend and the respective gradients
            const float3 color_pixel_w_bg = make_float3(
                image[pixel_idx],
                image[n_pixels + pixel_idx],
                image[2 * n_pixels + pixel_idx]
            );
            const float final_transmittance = tile_final_transmittances[pixel_idx];
            color_pixel_residual = color_pixel_w_bg - final_transmittance * background;
            // color and alpha gradients
            grad_color_pixel = make_float3(
                grad_image[pixel_idx],
                grad_image[n_pixels + pixel_idx],
                grad_image[2 * n_pixels + pixel_idx]
            );
            grad_alpha_common = final_transmittance * -dot(grad_color_pixel, background);
        }
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_idx[thread_rank] = primitive_idx;
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                collected_color[thread_rank] = primitive_color[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; j < current_batch_size; ++j) {
                float grad_color_x_sum = 0.0f;
                float grad_color_y_sum = 0.0f;
                float grad_color_z_sum = 0.0f;
                float grad_opacity_sum = 0.0f;
                float grad_conic_x_sum = 0.0f;
                float grad_conic_y_sum = 0.0f;
                float grad_conic_z_sum = 0.0f;
                float grad_mean_x_sum = 0.0f;
                float grad_mean_y_sum = 0.0f;
                float grad_mean_abs_x_sum = 0.0f;
                float grad_mean_abs_y_sum = 0.0f;
                bool active = !done;
                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                if (!config::original_stability_measures) exponent = fminf(exponent, 0.0f);
                else if (exponent > 0.0f) active = false;
                const float gaussian = expf(exponent);
                const float fragment_alpha = opacity * gaussian;
                if (fragment_alpha < config::min_alpha_threshold) active = false;
                const float alpha = config::original_stability_measures ? fminf(fragment_alpha, config::max_fragment_alpha) : fragment_alpha;

                // compute remaining transmittance after this fragment
                const float one_minus_alpha = 1.0f - alpha;
                const float next_transmittance = transmittance * one_minus_alpha;

                // early stopping as in original 3DGS, i.e., before blending (if config::original_stability_measures)
                if (active && config::original_stability_measures && next_transmittance < config::transmittance_threshold) {
                    done = true;
                    active = false;
                }

                // blending weight
                const float blending_weight = transmittance * alpha;
                const uint primitive_idx = collected_primitive_idx[j];
                const float3 color_unclamped = collected_color[j];

                if (active) {
                    // color gradient
                    const float3 dL_dcolor = blending_weight * grad_color_pixel;
                    grad_color_x_sum = color_unclamped.x >= 0.0f ? dL_dcolor.x : 0.0f;
                    grad_color_y_sum = color_unclamped.y >= 0.0f ? dL_dcolor.y : 0.0f;
                    grad_color_z_sum = color_unclamped.z >= 0.0f ? dL_dcolor.z : 0.0f;

                    const float3 color = fmaxf(color_unclamped, 0.0f);
                    color_pixel_residual -= blending_weight * color;

                    // alpha gradient
                    const float one_minus_alpha_rcp = 1.0f / (config::original_stability_measures ? one_minus_alpha : fmaxf(one_minus_alpha, config::one_minus_alpha_eps));
                    const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_residual * one_minus_alpha_rcp, grad_color_pixel);
                    const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
                    const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;

                    // opacity gradient
                    grad_opacity_sum = gaussian * dL_dalpha;

                    // conic and mean2d gradient
                    const float gaussian_grad_helper = -alpha * dL_dalpha;
                    const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(
                        delta.x * delta.x,
                        delta.x * delta.y,
                        delta.y * delta.y
                    );
                    grad_conic_x_sum = dL_dconic.x;
                    grad_conic_y_sum = dL_dconic.y;
                    grad_conic_z_sum = dL_dconic.z;
                    const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                        conic.x * delta.x + conic.y * delta.y,
                        conic.y * delta.x + conic.z * delta.y
                    );
                    grad_mean_x_sum = dL_dmean2d.x;
                    grad_mean_y_sum = dL_dmean2d.y;
                    grad_mean_abs_x_sum = fabsf(dL_dmean2d.x);
                    grad_mean_abs_y_sum = fabsf(dL_dmean2d.y);

                    // update transmittance
                    transmittance = next_transmittance;

                    // early stopping (if not config::original_stability_measures)
                    if (!config::original_stability_measures && transmittance < config::transmittance_threshold) {
                        done = true;
                    }
                }

                const GradientSums gradient_sums = {
                    grad_color_x_sum,
                    grad_color_y_sum,
                    grad_color_z_sum,
                    grad_opacity_sum,
                    grad_conic_x_sum,
                    grad_conic_y_sum,
                    grad_conic_z_sum,
                    grad_mean_x_sum,
                    grad_mean_y_sum,
                    grad_mean_abs_x_sum,
                    grad_mean_abs_y_sum
                };
                const GradientSums gradient_totals = GradientBlockReduce(gradient_reduce_storage).Reduce(gradient_sums, GradientSumsAdd());
                block.sync();

                if (thread_rank == 0) {
                    if (gradient_totals.color_x != 0.0f) atomicAdd(&grad_colors[primitive_idx], gradient_totals.color_x);
                    if (gradient_totals.color_y != 0.0f) atomicAdd(&grad_colors[n_primitives + primitive_idx], gradient_totals.color_y);
                    if (gradient_totals.color_z != 0.0f) atomicAdd(&grad_colors[2 * n_primitives + primitive_idx], gradient_totals.color_z);
                    if (gradient_totals.opacity != 0.0f) atomicAdd(&grad_opacity[primitive_idx], gradient_totals.opacity);
                    if (gradient_totals.conic_x != 0.0f) atomicAdd(&grad_conic[primitive_idx], gradient_totals.conic_x);
                    if (gradient_totals.conic_y != 0.0f) atomicAdd(&grad_conic[n_primitives + primitive_idx], gradient_totals.conic_y);
                    if (gradient_totals.conic_z != 0.0f) atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], gradient_totals.conic_z);
                    if (gradient_totals.mean_x != 0.0f) atomicAdd(&grad_mean2d[primitive_idx].x, gradient_totals.mean_x);
                    if (gradient_totals.mean_y != 0.0f) atomicAdd(&grad_mean2d[primitive_idx].y, gradient_totals.mean_y);
                    if (gradient_totals.mean_abs_x != 0.0f) atomicAdd(&grad_mean2d_abs[primitive_idx].x, gradient_totals.mean_abs_x);
                    if (gradient_totals.mean_abs_y != 0.0f) atomicAdd(&grad_mean2d_abs[primitive_idx].y, gradient_totals.mean_abs_y);
                }
            }
        }
    }

}
