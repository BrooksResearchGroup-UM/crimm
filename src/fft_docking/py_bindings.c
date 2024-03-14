#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include "fft_correlate.h"
#include "rank_poses.h"

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
    int roll_steps;
    if (!PyArg_ParseTuple(
      args, "O!iO!", 
      &PyArray_Type, &grids, 
      &roll_steps,
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

    sum_grids(
      grids, result, roll_steps
    );

    Py_RETURN_NONE;
}

// Function to pack the top scores and their indices into a tuple to return to Python
static PyObject *pack_top_scores_as_tuple(
  OrienPoseScore *top_scores, const int top_n_scores
  ){
    // Create numpy arrays for scores and indices
    npy_intp dims[1] = {top_n_scores};
    PyObject *out_scores = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *pose_ids = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject *orientation_ids = PyArray_SimpleNew(1, dims, NPY_INT);

    // Fill the numpy arrays
    for (int i = 0; i < top_n_scores; i++) {
      *(double *)PyArray_GETPTR1((PyArrayObject *)out_scores, i) = top_scores[i].score;
      *(int *)PyArray_GETPTR1((PyArrayObject *)pose_ids, i) = top_scores[i].pose_id;
      *(int *)PyArray_GETPTR1((PyArrayObject *)orientation_ids, i) = top_scores[i].orientation_id;
    }

    // Pack the numpy arrays into a tuple
    PyObject *result = PyTuple_Pack(3, out_scores, pose_ids, orientation_ids);

    // Decrease reference to the numpy arrays as they're now owned by the tuple
    Py_DECREF(out_scores);
    Py_DECREF(pose_ids);
    Py_DECREF(orientation_ids);

    return result;
}

static PyObject* py_rank_poses(PyObject* self, PyObject* args) {
    PyArrayObject *scores;
    int top_n_poses, sample_factor, n_threads, n_orientations, n_scores;
    if (!PyArg_ParseTuple(
      args, "O!iii", 
      &PyArray_Type, &scores, 
      &top_n_poses, &sample_factor, &n_threads
    )) {
        return NULL;
    }

    if (PyArray_TYPE(scores) != NPY_FLOAT32){
      PyErr_SetString(
        PyExc_TypeError, "Expected scores array of float32."
      );
      return NULL;
    }

    // Check array dimensions and data type
    if (PyArray_NDIM(scores) == 4) {
      // Get array dimensions
      n_orientations = PyArray_DIMS(scores)[0];
      int nx = PyArray_DIMS(scores)[1];
      int ny = PyArray_DIMS(scores)[2];
      int nz = PyArray_DIMS(scores)[3];
      n_scores = nx * ny * nz;
    } else if (PyArray_NDIM(scores) == 2) {
      // Get array dimensions
      n_orientations = PyArray_DIMS(scores)[0];
      n_scores = PyArray_DIMS(scores)[1];
    } else {
      PyErr_SetString(
        PyExc_TypeError, 
        "Expected scores array of float32 with 4 dimension \
        (n_orientations, nx, ny, nz), or 2 dimension (n_orientations, n_scores)."
      );
      return NULL;
    }
    // Allocate memory for the output array
    OrienPoseScore *top_scores = (OrienPoseScore *)malloc(
      top_n_poses * sizeof(OrienPoseScore)
    );
    float *scores_arr = (float *)PyArray_GETPTR1(scores, 0);
    rank_poses(
      scores_arr, n_orientations, n_scores, sample_factor, 
      top_n_poses, n_threads, top_scores
    );
    PyObject *result = pack_top_scores_as_tuple(top_scores, top_n_poses);
    return result;
}

// Method table
static PyMethodDef FftCorrelateMethods[] = {
    {"fft_correlate_batch", py_fft_correlate_batch, METH_VARARGS, "FFT Correlation Batch"},
    {"sum_grids", py_sum_grids, METH_VARARGS, "Sum grids"},
    {"rank_poses", py_rank_poses, METH_VARARGS, "Rank poses"},
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