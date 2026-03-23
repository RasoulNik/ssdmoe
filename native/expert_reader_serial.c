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

    for (int32_t i = 0; i < num_experts; ++i) {
        off_t offset = (off_t)abs_offset + (off_t)expert_indices[i] * (off_t)expert_stride;
        uint8_t *dst = out_buffer + ((uint64_t)i * expert_size);
        ssize_t n = pread(fd, dst, (size_t)expert_size, offset);
        if (n != (ssize_t)expert_size) {
            return -2;
        }
    }

    return 0;
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

    for (int32_t component_index = 0; component_index < num_components; ++component_index) {
        const int fd = fds[component_index];
        const uint64_t expert_size = expert_sizes[component_index];
        const uint64_t out_offset = component_output_offsets[component_index];
        for (int32_t expert_slot = 0; expert_slot < num_experts; ++expert_slot) {
            const off_t offset =
                (off_t)abs_offsets[component_index] +
                (off_t)expert_indices[expert_slot] * (off_t)expert_strides[component_index];
            uint8_t *dst =
                out_buffer +
                out_offset +
                ((uint64_t)expert_slot * expert_size);
            const ssize_t n = pread(fd, dst, (size_t)expert_size, offset);
            if (n != (ssize_t)expert_size) {
                return -2;
            }
        }
    }

    return 0;
}
