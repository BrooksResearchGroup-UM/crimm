#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#include <fftw3.h>
#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

// Function to fill indices array with integers from 0 to n
void fill_indices(int *indices, int n) {
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
}

// Function to swap two integers
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function to partition the indices array and return the pivot index
int partition(float *arr, int *indices, int low, int high) {
    float pivot = arr[indices[high]];
    int i = low - 1;
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[indices[j]] < pivot) {
            i++;
            swap(&indices[i], &indices[j]);
        }
    }
    swap(&indices[i + 1], &indices[high]);
    return i + 1;
}

// Quicksort function for argsort
void quicksort(float *arr, int *indices, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, indices, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Recursively sort elements before the pivot
                quicksort(arr, indices, low, pivot - 1);
            }
            #pragma omp section
            {
                // Recursively sort elements after the pivot
                quicksort(arr, indices, pivot + 1, high);
            }
        }
    }
}

// Function to perform batch 3D FFT correlation
void fft_correlate_batch(
  PyArrayObject *recep_grid, PyArrayObject *result, int n_threads
) {
  // recep_grid: (n_grids, nx, ny, nz)
  // result: (n_orientations, n_grids, nx, ny, nz)

  // Get array dimensions
  int n_grids = PyArray_DIMS(recep_grid)[0];
  int n_orientations = PyArray_DIMS(result)[0];
  int nx = PyArray_DIMS(recep_grid)[1];
  int ny = PyArray_DIMS(recep_grid)[2];
  int nz = PyArray_DIMS(recep_grid)[3];
  int N_grid_points = nx * ny * nz;
  // Number FFT coefficients (only half of the array is needed due to symmetry)
  int N_fft_points = nx * ny * (nz / 2 + 1);

  // Get NumPy arrays data pointers
  float *recep_arr = (float *)PyArray_GETPTR1(recep_grid, 0);
  float *lig_arr = (float *)PyArray_GETPTR1(result, 0);

  // Allocate memory for FFTW plans and data
  fftwf_plan plan_fwd, plan_inv;
  fftwf_complex *fft_r, *fft_l;
  fft_r = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_l = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points * n_orientations);

  // fftwf_plan_with_nthreads(n_threads);
  printf("Number of threads: %d\n", n_threads);

  // Create forward and inverse FFT plans for both arrays
  plan_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, recep_arr, fft_r, FFTW_MEASURE);
  plan_inv = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_l, lig_arr, FFTW_MEASURE);

  float scale = 1.0 / N_grid_points;
  // Execute forward FFTs on both arrays
  for (int i = 0; i < n_grids; i++) {
    float *cur_recep = (float *)PyArray_GETPTR1(recep_grid, i);
    fftwf_execute_dft_r2c(plan_fwd, cur_recep, fft_r);
    #pragma omp parallel for num_threads(n_threads)
    for (int j = 0; j < n_orientations; j++) {
      float *cur_lig = (float *)PyArray_GETPTR2(result, j, i);
      fftwf_complex *cur_fft_l = fft_l + j * N_fft_points;
      // The pointer to the padded ligand array is the same as the pointer 
      // to the correlation array
      fftwf_execute_dft_r2c(plan_fwd, cur_lig, cur_fft_l);
      // Perform element-wise complex conjugate multiplication
      for (int k = 0; k < N_fft_points; k++) {
        cur_fft_l[k] = conjf(fft_r[k]) * cur_fft_l[k] * scale;
      }
      // Execute inverse FFT on the product
      fftwf_execute_dft_c2r(plan_inv, cur_fft_l, cur_lig);
    }
  }

  // Clean up memory and plans
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
}

static PyObject *argsort_wrapper(PyObject *self, PyObject *args) {
    PyArrayObject *arr_in;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr_in)) {
        return NULL;
    }

    // Check if input array is a 1D array
    if (PyArray_NDIM(arr_in) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 1D");
        return NULL;
    }

    // Get input array dimensions
    npy_intp n = PyArray_DIM(arr_in, 0);

    // Get input array data
    float *arr = (float *)PyArray_DATA(arr_in);

    // Allocate memory for indices
    int *indices = (int *)malloc(n * sizeof(int));
    fill_indices(indices, n);

    // Perform argsort
    quicksort(arr, indices, 0, n - 1);

    // Create output array
    PyObject *indices_out = PyArray_SimpleNewFromData(1, &n, NPY_INT, indices);

    // Free memory
    // free(indices);

    return indices_out;
}

void sum_grids(
  PyArrayObject *grids, PyArrayObject *result
) {
  // Get array dimensions
  int n_orientations = PyArray_DIMS(grids)[0];
  int n_grids = PyArray_DIMS(grids)[1];
  int nx = PyArray_DIMS(grids)[2];
  int ny = PyArray_DIMS(grids)[3];
  int nz = PyArray_DIMS(grids)[4];
  int N_grid_points = nx * ny * nz;

  // Get NumPy arrays data pointers
  float *arr_grids = (float *)PyArray_DATA(grids);
  float *arr_result = (float *)PyArray_DATA(result);

  // Sum all grids
  #pragma omp parallel for
  for (int i = 0; i < n_orientations; i++) {
    for (int j = 0; j < n_grids; j++) {
      float *cur_arr_grids = arr_grids + (i * n_grids + j) * N_grid_points;
      float *cur_arr_result = arr_result + i * N_grid_points;
      for (int k = 0; k < N_grid_points; k++) {
        cur_arr_result[k] += cur_arr_grids[k];
      }
    }
  }
}

// Wrapper function to be called from Python
static PyObject* py_fft_correlate_batch(PyObject* self, PyObject* args) {
    PyArrayObject *recep_grid, *result;
    int n_threads;
    if (!PyArg_ParseTuple(
      args, "O!O!i", 
      &PyArray_Type, &recep_grid, 
      &PyArray_Type, &result,
      &n_threads
    )) {
        return NULL;
    }

    // Check array dimensions and data type
    if (PyArray_NDIM(recep_grid) != 4 || PyArray_TYPE(recep_grid) != NPY_FLOAT32) {
      PyErr_SetString(
        PyExc_TypeError, "Expected receptor arrays of float32 with 4 dimensions."
      );
      return NULL;
    }

    if (PyArray_NDIM(result) != 5 || PyArray_TYPE(result) != NPY_FLOAT32) {
      PyErr_SetString(
        PyExc_TypeError, "Expected result arrays of float32 with 5 dimensions."
      );
      return NULL;
    }
    int n_grids = PyArray_DIMS(recep_grid)[0];
    int n_grids_result = PyArray_DIMS(result)[1];
    if (n_grids != n_grids_result) {
      PyErr_SetString(
        PyExc_TypeError, 
        "Expected same number of grids for both receptor and result arrays."
      );
      return NULL;
    }

    fft_correlate_batch(recep_grid, result, n_threads);

    Py_RETURN_NONE;
}

static PyObject* py_sum_grids(PyObject* self, PyObject* args) {
    PyArrayObject *grids, *result;
    if (!PyArg_ParseTuple(
      args, "O!O!", 
      &PyArray_Type, &grids, 
      &PyArray_Type, &result
    )) {
        return NULL;
    }

    // Check array dimensions and data type
    if (PyArray_NDIM(grids) != 5 || PyArray_TYPE(grids) != NPY_FLOAT32) {
      PyErr_SetString(
        PyExc_TypeError, "Expected grids array of float32 with 5 dimensions."
      );
      return NULL;
    }

    if (PyArray_NDIM(result) != 4 || PyArray_TYPE(result) != NPY_FLOAT32) {
      PyErr_SetString(
        PyExc_TypeError, "Expected result array of float32 with 4 dimensions."
      );
      return NULL;
    }

    sum_grids(grids, result);

    Py_RETURN_NONE;
}

// Method table
static PyMethodDef FftCorrelateMethods[] = {
    {"fft_correlate_batch", py_fft_correlate_batch, METH_VARARGS, "FFT Correlation Batch"},
    {"sum_grids", py_sum_grids, METH_VARARGS, "Sum grids"},
    {"argsort", argsort_wrapper, METH_VARARGS, "Perform argsort"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef fftcorrelatemodule = {
    PyModuleDef_HEAD_INIT,
    "fft_correlate",   // name of module
    NULL,              // module documentation, may be NULL
    -1,                // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    FftCorrelateMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_fft_correlate(void) {
    import_array();
    return PyModule_Create(&fftcorrelatemodule);
}