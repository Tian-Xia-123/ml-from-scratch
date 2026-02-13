#define _POSIX_C_SOURCE 199309L
#include "tinynumpy.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int N = 10000000;
  int shape[] = {N};
  int ndim = 1;

  // Allocate arrays
  ndarray *a = np_array(ndim, shape);
  ndarray *b = np_array(ndim, shape);
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays initialization failed!\n");
    return 1;
  }

  // Initialize data
  for (int i = 0; i < N; i++) {
    a->data[i] = 1.0;
    b->data[i] = 2.0;
  }

  // Warm up the CPU cache and trigger lazy memory allocation
  ndarray *dummy = np_add(a, b);
  np_free(dummy);

  // Core benchmark logic
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start); // Get start time
  ndarray *res = np_add(a, b);
  clock_gettime(CLOCK_MONOTONIC, &end);

  // Calculate elapsed time
  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Array Size: %d\n", N);
  printf("tinynumpy np_add absolute latency: %f seconds\n", time_taken);
  printf("Computational Throughput: %f MB/s\n",
         (N * sizeof(double) * 3) / (time_taken * 1024 * 1024));

  // Cleanup
  np_free(a);
  np_free(b);
  np_free(res);

  return 0;
}