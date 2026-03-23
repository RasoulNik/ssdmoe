#include "expert_reader.h"
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int read_component_batch(
    int fd,
    uint64_t abs_offset,
    uint64_t expert_stride,
    uint64_t expert_size,
    const int32_t *expert_indices,
    int32_t num_experts,
    uint8_t *out_buffer
) {
    if (fd < 0 || expert_indices == NULL || out_buffer == NULL || num_experts <= 0) {
        return -1;
    }

    __block int failed = 0;
    dispatch_apply((size_t)num_experts, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        if (failed) return;
        off_t offset = (off_t)abs_offset + (off_t)expert_indices[i] * (off_t)expert_stride;
        uint8_t *dst = out_buffer + (i * expert_size);
        ssize_t n = pread(fd, dst, (size_t)expert_size, offset);
        if (n != (ssize_t)expert_size) {
            failed = 1;
        }
    });

    return failed ? -2 : 0;
}

/* Tasks <= this threshold run in a sequential loop to avoid GCD thread-wakeup
   overhead (~150 µs/call).  Above the threshold dispatch_apply parallelism
   outweighs its setup cost.

   Decode with no cache:   num_components(3) × K(4) = 12  → dispatch_apply  (unchanged)
   Decode with cache miss: num_components(3) × 1-2  = 3-6 → sequential loop (no GCD)
   Prefill:                num_components(3) × 30+  = 90+ → dispatch_apply  (unchanged)
*/
#define SEQUENTIAL_TASK_THRESHOLD 8

int read_component_batches(
    int32_t num_components,
    const int *fds,
    const uint64_t *abs_offsets,
    const uint64_t *expert_strides,
    const uint64_t *expert_sizes,
    const uint64_t *component_output_offsets,
    const int32_t *expert_indices,
    int32_t num_experts,
    uint8_t *out_buffer
) {
    if (num_components <= 0 || num_experts <= 0 || fds == NULL || abs_offsets == NULL ||
        expert_strides == NULL || expert_sizes == NULL || component_output_offsets == NULL ||
        expert_indices == NULL || out_buffer == NULL) {
        return -1;
    }

    const size_t total_tasks = (size_t)num_components * (size_t)num_experts;

    if (total_tasks <= SEQUENTIAL_TASK_THRESHOLD) {
        /* Sequential loop: avoids GCD thread-wakeup overhead for small batches. */
        for (int32_t c = 0; c < num_components; c++) {
            const uint64_t expert_size = expert_sizes[c];
            for (int32_t e = 0; e < num_experts; e++) {
                const off_t offset =
                    (off_t)abs_offsets[c] +
                    (off_t)expert_indices[e] * (off_t)expert_strides[c];
                uint8_t *dst =
                    out_buffer + component_output_offsets[c] + ((uint64_t)e * expert_size);
                ssize_t n = pread(fds[c], dst, (size_t)expert_size, offset);
                if (n != (ssize_t)expert_size) return -2;
            }
        }
        return 0;
    }

    __block int failed = 0;
    dispatch_apply(total_tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t task_index) {
        if (failed) return;
        const int32_t component_index = (int32_t)(task_index / (size_t)num_experts);
        const int32_t expert_slot     = (int32_t)(task_index % (size_t)num_experts);
        const int fd                  = fds[component_index];
        const uint64_t expert_size    = expert_sizes[component_index];
        const off_t offset =
            (off_t)abs_offsets[component_index] +
            (off_t)expert_indices[expert_slot] * (off_t)expert_strides[component_index];
        uint8_t *dst =
            out_buffer +
            component_output_offsets[component_index] +
            ((uint64_t)expert_slot * expert_size);
        const ssize_t n = pread(fd, dst, (size_t)expert_size, offset);
        if (n != (ssize_t)expert_size) {
            failed = 1;
        }
    });

    return failed ? -2 : 0;
}

/* Copy expert data from mmap regions using dispatch_apply (or sequential loop
   for small batches).  Identical contract to read_component_batches but reads
   from pre-mapped memory instead of issuing pread syscalls.  This lets the OS
   page cache serve warm expert bytes without any kernel-mode transition.

   mmap_bases[c] — start of the mmap region for component c's shard file.
   abs_offsets[c] — byte offset of expert[0] within that shard (same semantics
                    as in read_component_batches).
*/
int copy_component_batches_mmap(
    int32_t num_components,
    const void *const *mmap_bases,
    const uint64_t *abs_offsets,
    const uint64_t *expert_strides,
    const uint64_t *expert_sizes,
    const uint64_t *component_output_offsets,
    const int32_t *expert_indices,
    int32_t num_experts,
    uint8_t *out_buffer
) {
    if (num_components <= 0 || num_experts <= 0 || mmap_bases == NULL ||
        abs_offsets == NULL || expert_strides == NULL || expert_sizes == NULL ||
        component_output_offsets == NULL || expert_indices == NULL || out_buffer == NULL) {
        return -1;
    }

    const size_t total_tasks = (size_t)num_components * (size_t)num_experts;

    if (total_tasks <= SEQUENTIAL_TASK_THRESHOLD) {
        for (int32_t c = 0; c < num_components; c++) {
            const uint64_t expert_size = expert_sizes[c];
            const uint8_t *base = (const uint8_t *)mmap_bases[c] + abs_offsets[c];
            for (int32_t e = 0; e < num_experts; e++) {
                const uint8_t *src = base + (uint64_t)expert_indices[e] * expert_strides[c];
                uint8_t *dst =
                    out_buffer + component_output_offsets[c] + ((uint64_t)e * expert_size);
                memcpy(dst, src, (size_t)expert_size);
            }
        }
        return 0;
    }

    dispatch_apply(total_tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t task_index) {
        const int32_t component_index = (int32_t)(task_index / (size_t)num_experts);
        const int32_t expert_slot     = (int32_t)(task_index % (size_t)num_experts);
        const uint64_t expert_size    = expert_sizes[component_index];
        const uint8_t *src =
            (const uint8_t *)mmap_bases[component_index] +
            abs_offsets[component_index] +
            (uint64_t)expert_indices[expert_slot] * expert_strides[component_index];
        uint8_t *dst =
            out_buffer +
            component_output_offsets[component_index] +
            ((uint64_t)expert_slot * expert_size);
        memcpy(dst, src, (size_t)expert_size);
    });

    return 0;
}

int read_component_batches_into_slots(
    int32_t num_components,
    const int *fds,
    const uint64_t *abs_offsets,
    const uint64_t *expert_strides,
    const uint64_t *expert_sizes,
    void *const *out_buffers,
    const int32_t *expert_indices,
    const int32_t *slot_indices,
    int32_t num_experts
) {
    if (num_components <= 0 || num_experts <= 0 || fds == NULL || abs_offsets == NULL ||
        expert_strides == NULL || expert_sizes == NULL || out_buffers == NULL ||
        expert_indices == NULL || slot_indices == NULL) {
        return -1;
    }

    const size_t total_tasks = (size_t)num_components * (size_t)num_experts;
    __block int failed = 0;
    dispatch_apply(total_tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t task_index) {
        if (failed) return;
        const int32_t component_index = (int32_t)(task_index / (size_t)num_experts);
        const int32_t expert_slot     = (int32_t)(task_index % (size_t)num_experts);
        const int fd                  = fds[component_index];
        const uint64_t expert_size    = expert_sizes[component_index];
        const off_t offset =
            (off_t)abs_offsets[component_index] +
            (off_t)expert_indices[expert_slot] * (off_t)expert_strides[component_index];
        uint8_t *dst =
            (uint8_t *)out_buffers[component_index] +
            ((uint64_t)slot_indices[expert_slot] * expert_size);
        const ssize_t n = pread(fd, dst, (size_t)expert_size, offset);
        if (n != (ssize_t)expert_size) {
            failed = 1;
        }
    });

    return failed ? -2 : 0;
}
