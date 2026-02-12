#ifndef TINYNUMPY_H
#define TINYNUMPY_H

/**
 * Core data structure: ndarray
 * Represents a multi-dimensional array with contiguous memory layout.
 */

typedef struct {
  double *data; // Pointer to the raw data (stored as a flattened 1D array)
  int *shape;   // Size of each dimension
  int *strides; // Number of elements to skip to reach the next element
  int ndim;     // Number of dimensions
  int size;     // Total number of elements
} ndarray;

/* --- Function Prototypes --- */

// Creation and Destruction
/**
 * Allocates and initializes a new ndarray with the given dimensions.
 */
ndarray *np_array(int ndim, int *shape);

/**
 * Frees the memory associated with the ndarray, including data, shape, and
 * strides.
 */
void np_free(ndarray *arr);

// Data Access
/**
 * Retrieves the value at the specified multi-dimensional index.
 */
double np_get(ndarray *arr, int *indices);

/**
 * Sets the value at the specified multi-dimensional index.
 */
void np_set(ndarray *arr, int *indices, double value);

// Basic Operations
/**
 * Performs element-wise addition of two arrays.
 * Note: Assumes both arrays have the same shape.
 */
ndarray *np_add(ndarray *a, ndarray *b);

#endif