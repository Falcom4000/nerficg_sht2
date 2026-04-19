#include "metric_counts.h"
#include "pruning_scores.h"
#include "kernels_metric_counts.cuh"
#include "kernels_instances.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

void fast_gs::rasterization::metric_counts(
    std::function<char* (size_t)> resize_primitive_buffers,
    std::function<char* (size_t)> resize_tile_buffers,
    std::function<char* (size_t)> resize_instance_buffers,
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_coefficients_0,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const int* metric_map,
    float* counts,
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    const dim3 grid(div_round_up(width,  config::tile_width),
                    div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    char* tile_buffers_blob = resize_tile_buffers(required<TileBuffers>(n_tiles));
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(tile_buffers.instance_ranges, 0,
                        sizeof(uint2) * n_tiles, memset_stream);
    } else {
        cudaMemset(tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);
    }

    PrimitiveBuffers primitive_buffers;
    int n_visible_primitives, n_instances;
    run_preprocessing(
        resize_primitive_buffers,
        means, scales, rotations, opacities,
        sh_coefficients_0, sh_coefficients_rest,
        w2c, cam_position,
        primitive_buffers, grid, n_primitives,
        active_sh_bases, total_sh_bases,
        width, height, focal_x, focal_y, center_x, center_y,
        near_plane, far_plane, proper_antialiasing,
        n_visible_primitives, n_instances);

    #define COMPUTE_METRIC_COUNTS_ARGS \
        resize_instance_buffers, \
        primitive_buffers, \
        tile_buffers, \
        grid, block, \
        metric_map, counts, \
        memset_stream, \
        n_visible_primitives, n_instances, end_bit, \
        width, height
    if (end_bit <= 16) compute_metric_counts<ushort>(COMPUTE_METRIC_COUNTS_ARGS);
    else               compute_metric_counts<uint>(COMPUTE_METRIC_COUNTS_ARGS);
    #undef COMPUTE_METRIC_COUNTS_ARGS
}

template <typename KeyT>
void fast_gs::rasterization::compute_metric_counts(
    std::function<char* (size_t)>& resize_instance_buffers,
    PrimitiveBuffers& primitive_buffers,
    TileBuffers& tile_buffers,
    const dim3& grid,
    const dim3& block,
    const int* metric_map,
    float* counts,
    const cudaStream_t memset_stream,
    const int n_visible_primitives,
    const int n_instances,
    const int end_bit,
    const int width,
    const int height)
{
    char* instance_buffers_blob =
        resize_instance_buffers(required<InstanceBuffers<KeyT>>(n_instances, end_bit));
    InstanceBuffers<KeyT> instance_buffers =
        InstanceBuffers<KeyT>::from_blob(instance_buffers_blob, n_instances, end_bit);

    kernels::pruning_scores::create_instances_cu<KeyT><<<
        div_round_up(n_visible_primitives, config::block_size_create_instances),
        config::block_size_create_instances>>>(
        primitive_buffers.primitive_indices.Current(),
        primitive_buffers.offset,
        primitive_buffers.screen_bounds,
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        instance_buffers.keys.Current(),
        instance_buffers.primitive_indices.Current(),
        grid.x, n_visible_primitives);
    CHECK_CUDA(config::debug, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        instance_buffers.cub_workspace, instance_buffers.cub_workspace_size,
        instance_buffers.keys, instance_buffers.primitive_indices,
        n_instances, 0, end_bit);
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (tile)")

    if constexpr (!config::debug) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::pruning_scores::extract_instance_ranges_cu<KeyT><<<
            div_round_up(n_instances, config::block_size_extract_instance_ranges),
            config::block_size_extract_instance_ranges>>>(
            instance_buffers.keys.Current(),
            tile_buffers.instance_ranges,
            n_instances);
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::metric_counts::compute_metric_counts_cu<<<grid, block>>>(
        tile_buffers.instance_ranges,
        instance_buffers.primitive_indices.Current(),
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        metric_map,
        counts,
        width, height, grid.x);
    CHECK_CUDA(config::debug, "compute_metric_counts")
}
