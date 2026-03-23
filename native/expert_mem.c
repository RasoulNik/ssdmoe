#include "expert_reader.h"
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void *alloc_buffer(uint64_t size) {
    if (size == 0) {
        return NULL;
    }
    return malloc((size_t)size);
}

void free_buffer(void *ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

void *alloc_slab(uint64_t total_size, uint64_t alignment) {
    if (total_size == 0 || alignment == 0) {
        return NULL;
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, (size_t)alignment, (size_t)total_size) != 0) {
        return NULL;
    }
    return ptr;
}

void free_slab(void *ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

int copy_experts_multi(
    int32_t num_components,
    const void *const *src_ptrs,
    void *const *dst_ptrs,
    const uint64_t *expert_sizes,
    const int32_t *src_slots,
    const int32_t *dst_slots,
    int32_t num_copies
) {
    if (num_components <= 0 || num_copies <= 0 ||
        src_ptrs == NULL || dst_ptrs == NULL ||
        expert_sizes == NULL || src_slots == NULL || dst_slots == NULL) {
        return -1;
    }

    const size_t total_tasks = (size_t)num_components * (size_t)num_copies;
    dispatch_apply(total_tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t task_index) {
        const int32_t c   = (int32_t)(task_index / (size_t)num_copies);
        const int32_t i   = (int32_t)(task_index % (size_t)num_copies);
        const uint64_t esz = expert_sizes[c];
        const uint8_t *src = (const uint8_t *)src_ptrs[c] + (uint64_t)src_slots[i] * esz;
        uint8_t       *dst = (uint8_t *)      dst_ptrs[c] + (uint64_t)dst_slots[i] * esz;
        memcpy(dst, src, (size_t)esz);
    });

    return 0;
}
