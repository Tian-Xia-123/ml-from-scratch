#include "tinynumpy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void np_free(ndarray *arr) {
  if (!arr) {
    return;
  }

  free(arr->data);
  free(arr->shape);
  free(arr->strides);
  free(arr);
}

np_status np_array(int32_t ndim, int64_t *shape, ndarray **out) {
  np_status status = NP_OK;

  if (!out || !shape) return NP_ERR_NULL_PTR;

  *out = NULL;

  if (ndim < 1 || ndim > MAX_DIMS) return NP_ERR_INVALID_DIM;

  *out = calloc(1, sizeof(**out));
  if (!*out) {
    status = NP_ERR_ALLOC;
    goto fail;
  }

  (*out)->ndim = ndim;
  (*out)->size = 1;

  (*out)->shape = malloc(ndim * sizeof(int64_t));
  if (!(*out)->shape) {
    status = NP_ERR_ALLOC;
    goto fail;
  }

  for (int32_t i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      status = NP_ERR_INVALID_SHAPE;
      goto fail;
    }
    (*out)->shape[i] = shape[i];

    if (shape[i] > INT64_MAX / (*out)->size) {
      status = NP_ERR_OVERFLOW;
      goto fail;
    }
    (*out)->size *= shape[i];
  }

  (*out)->strides = malloc(ndim * sizeof(int64_t));
  if (!(*out)->strides) {
    status = NP_ERR_ALLOC;
    goto fail;
  }

  int64_t current_stride = 1;
  for (int32_t i = ndim - 1; i >= 0; i--) {
    (*out)->strides[i] = current_stride;
    current_stride *= shape[i];
  }

  (*out)->data = calloc((*out)->size, sizeof(double));
  if (!(*out)->data) {
    status = NP_ERR_ALLOC;
    goto fail;
  }

  return status;

fail:
  NP_SAFE_FREE(*out);
  return status;
}

np_status np_zeros(int32_t ndim, int64_t *shape, ndarray **out) {
  return np_array(ndim, shape, out);
}

np_status np_ones(int32_t ndim, int64_t *shape, ndarray **out) {
  np_status status = np_array(ndim, shape, out);
  if (status != NP_OK) {
    return status;
  }

  for (int64_t i = 0; i < (*out)->size; i++) {
    (*out)->data[i] = 1.0;
  }
  return status;
}

np_status np_arange(double start, double stop, double step, ndarray **out) {
  if (!out) {
    return NP_ERR_NULL_PTR;
  }

  *out = NULL;

  if (step > 0 && start >= stop || step < 0 && start <= stop ||
      fabs(step) < 1e-10) {
    return NP_ERR_INVALID_STEP;
  }

  double epsilon = 1e-10;

  int64_t size = (int64_t)floor((stop - start - epsilon) / step) + 1;
  int64_t shape[1] = {size};

  np_status status = np_array(1, shape, out);
  if (status != NP_OK) {
    return status;
  }

  for (int64_t i = 0; i < size; i++) {
    (*out)->data[i] = start + i * step;
  }
  return status;
}

np_status np_get(ndarray *arr, int64_t *indices, double *out) {
  if (!arr || !indices || !out) {
    return NP_ERR_NULL_PTR;
  }

  int64_t offset = 0;
  for (int32_t i = 0; i < arr->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= arr->shape[i]) {
      return NP_ERR_OUT_OF_BOUNDS;
    }
    offset += indices[i] * arr->strides[i];
  }
  *out = arr->data[offset];

  return NP_OK;
}

np_status np_set(ndarray *arr, int64_t *indices, double value) {
  if (!arr || !indices) {
    return NP_ERR_NULL_PTR;
  }

  int64_t offset = 0;
  for (int32_t i = 0; i < arr->ndim; i++) {
    if (indices[i] < 0 || indices[i] >= arr->shape[i]) {
      return NP_ERR_OUT_OF_BOUNDS;
    }
    offset += indices[i] * arr->strides[i];
  }
  arr->data[offset] = value;

  return NP_OK;
}

np_status np_add(ndarray *a, ndarray *b, ndarray **out) {
  if (!out) {
    return NP_ERR_NULL_PTR;
  }

  *out = NULL;

  if (!a || !b) {
    return NP_ERR_NULL_PTR;
  }

  if (a->ndim != b->ndim) {
    return NP_ERR_DIM_MISMATCH;
  }

  for (int32_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      return NP_ERR_SHAPE_MISMATCH;
    }
  }

  np_status status = np_array(a->ndim, a->shape, out);
  if (status != NP_OK) {
    return status;
  }

  for (int64_t i = 0; i < a->size; i++) {
    (*out)->data[i] = a->data[i] + b->data[i];
  }
  return status;
}

np_status np_subtract(ndarray *a, ndarray *b, ndarray **out) {
  if (!out) {
    return NP_ERR_NULL_PTR;
  }

  *out = NULL;

  if (!a || !b) {
    return NP_ERR_NULL_PTR;
  }

  if (a->ndim != b->ndim) {
    return NP_ERR_DIM_MISMATCH;
  }

  for (int32_t i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i]) {
      return NP_ERR_SHAPE_MISMATCH;
    }
  }

  np_status status = np_array(a->ndim, a->shape, out);
  if (status != NP_OK) {
    return status;
  }

  for (int64_t i = 0; i < a->size; i++) {
    (*out)->data[i] = a->data[i] - b->data[i];
  }
  return status;
}

static void _matmul_2d_core(const double *A, const double *B, double *C,
                            int64_t M, int64_t K, int64_t N) {
  for (int64_t i = 0; i < M * N; i++) {
    C[i] = 0.0;
  }

  for (int64_t m = 0; m < M; m++) {
    for (int64_t k = 0; k < K; k++) {
      int64_t temp = m * K + k;
      for (int64_t n = 0; n < N; n++) {
        C[m * N + n] += A[temp] * B[k * N + n];
      }
    }
  }
}

np_status np_matmul(ndarray *a, ndarray *b, ndarray **out) {
  if (!out) {
    return NP_ERR_NULL_PTR;
  }

  *out = NULL;

  if (!a || !b) {
    return NP_ERR_NULL_PTR;
  }

  if (a->ndim != b->ndim) {
    return NP_ERR_DIM_MISMATCH;
  }

  if (a->shape[1] != b->shape[0]) {
    return NP_ERR_SHAPE_MISMATCH;
  }

  int64_t M = a->shape[0];
  int64_t K = a->shape[1];
  int64_t N = b->shape[1];

  int64_t shape[2] = {M, N};
  np_status status = np_array(2, shape, out);
  if (status != NP_OK) {
    return status;
  }

  _matmul_2d_core(a->data, b->data, (*out)->data, M, K, N);

  return status;
}