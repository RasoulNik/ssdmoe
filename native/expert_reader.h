#pragma once
#include <stdint.h>

/* I/O — pread-based batch loading */

int read_component_batch(
    int fd,
    uint64_t abs_offset,
    uint64_t expert_stride,
    uint64_t expert_size,
    const int32_t *expert_indices,
    int32_t num_experts,
    uint8_t *out_buffer);

int read_component_batches(
    int32_t num_components,
    const int *fds,
    const uint64_t *abs_offsets,
    const uint64_t *expert_strides,
    const uint64_t *expert_sizes,
    const uint64_t *component_output_offsets,
    const int32_t *expert_indices,
    int32_t num_experts,
    uint8_t *out_buffer);

int read_component_batches_into_slots(
    int32_t num_components,
    const int *fds,
    const uint64_t *abs_offsets,
    const uint64_t *expert_strides,
    const uint64_t *expert_sizes,
    void *const *out_buffers,
    const int32_t *expert_indices,
    const int32_t *slot_indices,
    int32_t num_experts);

/* Memory — aligned allocation and batch copy */

void *alloc_buffer(uint64_t size);
void  free_buffer(void *ptr);

/* Allocate a contiguous slab with the given alignment (must be a power of 2,
   >= sizeof(void*)).  Use free_slab to release. */
void *alloc_slab(uint64_t total_size, uint64_t alignment);
void  free_slab(void *ptr);

/* Copy num_copies experts across num_components in one dispatch_apply.
 *   src_ptrs[c] / dst_ptrs[c]  — base pointer for component c
 *   expert_sizes[c]             — bytes per expert for component c
 *   src_slots[i] / dst_slots[i] — slot index for expert i
 * Total tasks dispatched: num_components * num_copies
 */
int copy_experts_multi(
    int32_t num_components,
    const void *const *src_ptrs,
    void *const *dst_ptrs,
    const uint64_t *expert_sizes,
    const int32_t *src_slots,
    const int32_t *dst_slots,
    int32_t num_copies);
