#ifndef TINYNUMPY_H
#define TINYNUMPY_H

#include <stdint.h>

#define MAX_DIMS 32
#define NP_SAFE_FREE(ptr)                                                      \
  do {                                                                         \
    np_free(ptr);                                                              \
    (ptr) = NULL;                                                              \
  } while (0)

typedef enum {
  NP_OK = 0,
  NP_ERR_INVALID_DIM,
  NP_ERR_DIM_MISMATCH,
  NP_ERR_INVALID_SHAPE,
  NP_ERR_SHAPE_MISMATCH,
  NP_ERR_INVALID_STEP,
  NP_ERR_NULL_PTR,
  NP_ERR_ALLOC,
  NP_ERR_OVERFLOW,
  NP_ERR_OUT_OF_BOUNDS
} np_status;

typedef struct {
  double *data;
  int32_t ndim;
  int64_t *shape;
  int64_t *strides;
  int64_t size;
} ndarray;

void np_free(ndarray *arr);

np_status np_array(int32_t ndim, int64_t *shape, ndarray **out);

np_status np_zeros(int32_t ndim, int64_t *shape, ndarray **out);

np_status np_ones(int32_t ndim, int64_t *shape, ndarray **out);

np_status np_arange(double start, double stop, double step, ndarray **out);

np_status np_get(ndarray *arr, int64_t *indices, double *out);

np_status np_set(ndarray *arr, int64_t *indices, double value);

np_status np_add(ndarray *a, ndarray *b, ndarray **out);

np_status np_subtract(ndarray *a, ndarray *b, ndarray **out);

np_status np_matmul(ndarray *a, ndarray *b, ndarray **out);

#endif