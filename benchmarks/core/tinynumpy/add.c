#define _POSIX_C_SOURCE 199309L
#include "perf_monitor.h"
#include "tinynumpy.h"
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main() {
  int64_t N = 50000000;
  int64_t shape[] = {N};
  int32_t ndim = 1;

  // Initialize resources
  int fd_access = -1, fd_miss = -1;
  perf_result *access_res = malloc(sizeof(*access_res));
  perf_result *miss_res = malloc(sizeof(*miss_res));
  ndarray *a = NULL, *b = NULL, *dummy = NULL, *res = NULL;
  int ret = 0;

  // Setup performance monitoring events
  fd_access = setup_perf_event(PERF_TYPE_HW_CACHE,
                               PERF_COUNT_HW_CACHE_L1D |
                                   PERF_COUNT_HW_CACHE_OP_READ << 8 |
                                   PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
  fd_miss = setup_perf_event(PERF_TYPE_HW_CACHE,
                             PERF_COUNT_HW_CACHE_L1D |
                                 PERF_COUNT_HW_CACHE_OP_READ << 8 |
                                 PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

  if (fd_access == -1 || fd_miss == -1 || !access_res || !miss_res) {
    ret = -1;
    goto cleanup;
  }

  // Initialize arrays
  if (np_array(ndim, shape, &a) != NP_OK ||
      np_array(ndim, shape, &b) != NP_OK) {
    ret = -1;
    goto cleanup;
  }
  for (int64_t i = 0; i < N; i++) {
    a->data[i] = 1.0;
    b->data[i] = 2.0;
  }

  // Warm up the CPU cache and trigger lazy memory allocation
  np_add(a, b, &dummy);

  start_perf_event(fd_access);
  start_perf_event(fd_miss);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  if (np_add(a, b, &res) != NP_OK) {
    ret = -1;
    goto cleanup;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);

  stop_perf_event(fd_access);
  stop_perf_event(fd_miss);

  read_perf_event(fd_access, access_res);
  read_perf_event(fd_miss, miss_res);

  // Print results
  printf("Array Size: %" PRId64 "\n", N);
  printf("Cache L1D Accesses: %" PRIu64 ", Misses: %" PRIu64
         ", Miss Rate: %.2f%%\n",
         access_res->value, miss_res->value,
         (double)miss_res->value / access_res->value * 100.0);
  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Absolute latency: %f seconds\n", time_taken);
  printf("Computational Throughput: %f MB/s\n",
         (N * sizeof(double) * 3) / (time_taken * 1024 * 1024));

  goto cleanup;

cleanup:
  NP_SAFE_FREE(a);
  NP_SAFE_FREE(b);
  NP_SAFE_FREE(dummy);
  NP_SAFE_FREE(res);
  free(access_res);
  free(miss_res);
  if (fd_access != -1) close(fd_access);
  if (fd_miss != -1) close(fd_miss);
  return ret;
}