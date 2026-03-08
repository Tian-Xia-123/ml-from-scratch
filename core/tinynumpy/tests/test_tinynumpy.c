#include "tinynumpy.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define RUN_TEST(test_func)                                                    \
  do {                                                                         \
    printf("Running " #test_func "... ");                                      \
    test_func();                                                               \
    printf("\033[0;32m[PASS]\033[0m\n");                                       \
  } while (0)

void test_np_array() {
  int64_t shape[] = {2, 3};
  ndarray *a = NULL;
  np_status status = np_array(2, shape, &(a));

  assert(status == NP_OK);
  assert(a != NULL);
  assert(a->ndim == 2);
  assert(a->size == 6);
  assert(a->shape[0] == 2 && a->shape[1] == 3);

  NP_SAFE_FREE(a);
}

void test_np_zeros() {
  int64_t shape[] = {3, 4, 5};
  ndarray *a = NULL;
  np_status status = np_zeros(3, shape, &a);

  assert(status == NP_OK);
  assert(a != NULL);
  assert(a->ndim == 3);
  assert(a->size == 60);
  assert(a->shape[0] == 3 && a->shape[1] == 4 && a->shape[2] == 5);
  for (int64_t i = 0; i < a->size; i++) {
    assert(fabs(a->data[i]) < 1e-10);
  }

  NP_SAFE_FREE(a);
}

void test_np_ones() {
  int64_t shape[] = {2, 3, 6, 4};
  ndarray *a = NULL;
  np_status status = np_ones(4, shape, &a);

  assert(status == NP_OK);
  assert(a != NULL);
  assert(a->ndim == 4);
  assert(a->size == 144);
  assert(a->shape[0] == 2 && a->shape[1] == 3 && a->shape[2] == 6 &&
         a->shape[3] == 4);
  for (int64_t i = 0; i < a->size; i++) {
    assert(fabs(a->data[i] - 1.0) < 1e-10);
  }

  NP_SAFE_FREE(a);
}

void test_np_arange() {
  ndarray *a = NULL;
  np_status status = np_arange(0.0, 5.0, 1, &a);

  assert(status == NP_OK);
  assert(a != NULL);
  assert(a->ndim == 1);
  assert(a->size == 5);
  assert(a->shape[0] == 5);
  for (int64_t i = 0; i < a->size; i++) {
    assert(fabs(a->data[i] - i * 1.0) < 1e-10);
  }

  NP_SAFE_FREE(a);
}

void test_np_get() {
  int64_t shape[] = {2, 3, 6};
  ndarray *a = NULL;
  np_array(3, shape, &a);
  for (int64_t i = 0; i < a->size; i++) {
    a->data[i] = (double)i;
  }

  int64_t indices[3] = {1, 2, 3};
  double value;
  np_status status = np_get(a, indices, &value);
  assert(status == NP_OK);
  assert(fabs(value - 33.0) < 1e-10);

  NP_SAFE_FREE(a);
}

void test_np_set() {
  int64_t shape[] = {2, 3, 6};
  ndarray *a = NULL;
  np_array(3, shape, &a);

  int64_t indices[3] = {1, 2, 3};
  np_status status = np_set(a, indices, 6.5);
  assert(status == NP_OK);

  double value;
  np_get(a, indices, &value);
  assert(fabs(value - 6.5) < 1e-10);

  NP_SAFE_FREE(a);
}

void test_np_add() {
  int64_t shape[] = {6, 10, 5};
  ndarray *a = NULL;
  ndarray *b = NULL;
  np_array(3, shape, &a);
  np_array(3, shape, &b);
  for (int64_t i = 0; i < a->size; i++) {
    a->data[i] = 1.0;
    b->data[i] = 2.0;
  }

  ndarray *res = NULL;
  np_status status = np_add(a, b, &res);
  assert(status == NP_OK);
  for (int64_t i = 0; i < a->size; i++) {
    assert(fabs(res->data[i] - 3.0) < 1e-10);
  }

  NP_SAFE_FREE(a);
  NP_SAFE_FREE(b);
  NP_SAFE_FREE(res);
}

void test_np_subtract() {
  int64_t shape[] = {6, 10, 5};
  ndarray *a = NULL;
  ndarray *b = NULL;
  np_array(3, shape, &a);
  np_array(3, shape, &b);
  for (int i = 0; i < a->size; i++) {
    a->data[i] = 20.5;
    b->data[i] = 5.1;
  }

  ndarray *res = NULL;
  np_status status = np_subtract(a, b, &res);
  assert(status == NP_OK);
  for (int64_t i = 0; i < a->size; i++) {
    assert(fabs(res->data[i] - 15.4) < 1e-9);
  }

  NP_SAFE_FREE(a);
  NP_SAFE_FREE(b);
  NP_SAFE_FREE(res);
}

void test_np_matmul() {
  int64_t shape_a[2] = {2, 3};
  int64_t shape_b[2] = {3, 2};

  ndarray *a = NULL;
  ndarray *b = NULL;
  np_array(2, shape_a, &a);
  np_array(2, shape_b, &b);

  for (int64_t i = 0; i < a->size; i++) {
    a->data[i] = i + 1.0;
    b->data[i] = i + 7.0;
  }

  ndarray *res = NULL;
  np_status status = np_matmul(a, b, &res);
  assert(fabs(res->data[0] - 58.0) < 1e-10);
  assert(fabs(res->data[1] - 64.0) < 1e-10);
  assert(fabs(res->data[2] - 139.0) < 1e-10);
  assert(fabs(res->data[3] - 154.0) < 1e-10);

  NP_SAFE_FREE(a);
  NP_SAFE_FREE(b);
  NP_SAFE_FREE(res);
}

int main() {
  printf("\n--- Starting TinyNumPy Tests ---\n\n");

  RUN_TEST(test_np_array);
  RUN_TEST(test_np_zeros);
  RUN_TEST(test_np_ones);
  RUN_TEST(test_np_arange);
  RUN_TEST(test_np_get);
  RUN_TEST(test_np_set);
  RUN_TEST(test_np_add);
  RUN_TEST(test_np_subtract);
  RUN_TEST(test_np_matmul);

  printf("\n\033[1;32mAll tests completed successfully!\033[0m ✨\n");
  return 0;
}