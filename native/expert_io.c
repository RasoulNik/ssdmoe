#include "expert_reader.h"
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>
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
