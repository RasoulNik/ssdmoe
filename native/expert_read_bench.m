#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static NSArray<NSNumber *> *parseExperts(NSString *csv) {
    NSMutableArray<NSNumber *> *out = [NSMutableArray array];
    for (NSString *part in [csv componentsSeparatedByString:@","]) {
        NSString *trimmed = [part stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (trimmed.length == 0) continue;
        [out addObject:@(trimmed.intValue)];
    }
    return out;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: expert_read_bench <index.json> [layer] [experts_csv] [iters]\n");
            return 1;
        }

        NSString *indexPath = [NSString stringWithUTF8String:argv[1]];
        int layer = (argc >= 3) ? atoi(argv[2]) : 0;
        NSString *expertsCSV = (argc >= 4) ? [NSString stringWithUTF8String:argv[3]] : @"0,1,2,3,4,5,6,7";
        int iters = (argc >= 5) ? atoi(argv[4]) : 20;

        NSData *jsonData = [NSData dataWithContentsOfFile:indexPath];
        if (!jsonData) {
            fprintf(stderr, "failed to read %s\n", argv[1]);
            return 1;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&error];
        if (!root) {
            fprintf(stderr, "json parse error: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        NSString *modelPath = root[@"model_path"];
        NSDictionary *expertReads = root[@"expert_reads"][[NSString stringWithFormat:@"%d", layer]];
        if (!expertReads) {
            fprintf(stderr, "layer %d not found\n", layer);
            return 1;
        }

        NSMutableDictionary<NSString *, NSNumber *> *fds = [NSMutableDictionary dictionary];
        for (NSString *component in expertReads) {
            NSString *fileName = expertReads[component][@"file"];
            if (fds[fileName] != nil) continue;
            NSString *fullPath = [modelPath stringByAppendingPathComponent:fileName];
            int fd = open(fullPath.UTF8String, O_RDONLY);
            if (fd < 0) {
                perror(fullPath.UTF8String);
                return 1;
            }
            fds[fileName] = @(fd);
        }

        NSArray<NSString *> *components = [[expertReads allKeys] sortedArrayUsingSelector:@selector(compare:)];
        NSArray<NSNumber *> *experts = parseExperts(expertsCSV);
        __block uint64_t totalBytes = 0;

        NSMutableArray<NSNumber *> *samples = [NSMutableArray array];
        for (int iter = 0; iter < iters; iter++) {
            double t0 = now_ms();
            dispatch_group_t group = dispatch_group_create();

            for (NSNumber *expertNum in experts) {
                dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                    int expert = expertNum.intValue;
                    for (NSString *component in components) {
                        NSDictionary *info = expertReads[component];
                        NSString *fileName = info[@"file"];
                        int fd = fds[fileName].intValue;
                        off_t offset = [info[@"abs_offset"] longLongValue] +
                                       (off_t)expert * [info[@"expert_stride"] longLongValue];
                        size_t size = (size_t)[info[@"expert_size"] unsignedLongLongValue];
                        void *buf = malloc(size);
                        ssize_t n = pread(fd, buf, size, offset);
                        if (n > 0) {
                            __sync_fetch_and_add(&totalBytes, (uint64_t)n);
                        }
                        free(buf);
                    }
                });
            }

            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
            [samples addObject:@(now_ms() - t0)];
        }

        double meanMs = 0.0;
        double bestMs = DBL_MAX;
        for (NSNumber *sample in samples) {
            double v = sample.doubleValue;
            meanMs += v;
            if (v < bestMs) bestMs = v;
        }
        meanMs /= samples.count;

        double gib = (double)totalBytes / (1024.0 * 1024.0 * 1024.0);
        double meanSeconds = (meanMs / 1000.0) * samples.count;
        double gbps = gib / meanSeconds;

        printf("layer=%d experts=%s iters=%d\n", layer, expertsCSV.UTF8String, iters);
        printf("payload_total_gib=%.4f\n", gib);
        printf("mean_ms=%.2f\n", meanMs);
        printf("best_ms=%.2f\n", bestMs);
        printf("aggregate_gib_per_s=%.3f\n", gbps);

        for (NSString *fileName in fds) {
            close(fds[fileName].intValue);
        }
    }
    return 0;
}
