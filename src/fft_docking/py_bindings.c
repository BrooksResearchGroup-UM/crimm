#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include "fft_correlate.h"
#include "rank_poses.h"
#include "grid_gen.h"

// Wrapper function to be called from Python
// Grid generation functions
static PyObject* py_generate_grids(PyObject* self, PyObject* args) {
    PyArrayObject *grid_pos, *coords, *charges, *epsilons, *vdw_rs;
    float cc_elec, rad_dielec_const, elec_rep_max, elec_attr_max;
    float vwd_rep_max, vwd_attr_max;
    int use_constant_dielectric;
    if (!PyArg_ParseTuple(
        args, "O!O!O!O!O!ffffffi", 
        &PyArray_Type, &grid_pos,
        &PyArray_Type, &coords, 
        &PyArray_Type, &charges, 
        &PyArray_Type, &epsilons, 
        &PyArray_Type, &vdw_rs, 
        &cc_elec, &rad_dielec_const, &elec_rep_max, &elec_attr_max, 
        &vwd_rep_max, &vwd_attr_max, &use_constant_dielectric
    )) {
        return NULL;
    }
    // Check array dimensions and data type
    if (
        PyArray_TYPE(coords) != NPY_FLOAT32 || PyArray_TYPE(charges) != NPY_FLOAT32 || 
        PyArray_TYPE(epsilons) != NPY_FLOAT32 || PyArray_TYPE(vdw_rs) != NPY_FLOAT32 || 
        PyArray_TYPE(grid_pos) != NPY_FLOAT32
    ) {
        return PyErr_Format(
          PyExc_TypeError, "Expected arrays of float32 for the parameter arrays" 
        );
    }
    if (PyArray_NDIM(grid_pos) != 2) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected grid_pos array with 2 dimensions"
        );
        return NULL;
    }
    if (PyArray_NDIM(coords) != 2) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected coords array with 2 dimensions"
        );
        return NULL;
    }
    if (PyArray_NDIM(charges) != 1) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected charges array with 1 dimension"
        );
        return NULL;
    }
    if (PyArray_NDIM(epsilons) != 1) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected epsilons array with 1 dimension"
        );
        return NULL;
    }
    if (PyArray_NDIM(vdw_rs) != 1) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected vdw_rs array with 1 dimension"
        );
        return NULL;
    }
    
    int N_coords = PyArray_DIMS(coords)[0];
    int N_grid_points = PyArray_DIMS(grid_pos)[0];

    if (PyArray_DIMS(charges)[0] != N_coords) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected charges array with the same length as coords"
        );
        return NULL;
    }
    if (PyArray_DIMS(epsilons)[0] != N_coords) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected epsilons array with the same length as coords"
        );
        return NULL;
    }
    if (PyArray_DIMS(vdw_rs)[0] != N_coords) {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected vdw_rs array with the same length as coords"
        );
        return NULL;
    }
    // Get NumPy arrays data pointers
    float *coords_arr = (float *)PyArray_DATA(coords);
    float *charges_arr = (float *)PyArray_DATA(charges);
    float *epsilons_arr = (float *)PyArray_DATA(epsilons);
    float *vdw_rs_arr = (float *)PyArray_DATA(vdw_rs);
    float *grid_pos_arr = (float *)PyArray_DATA(grid_pos);
    // Allocate memory for the result arrays
    float *electrostat_grid = (float *)calloc(N_grid_points, sizeof(float));
    float *vdw_grid_attr = (float *)calloc(N_grid_points, sizeof(float));
    float *vdw_grid_rep = (float *)calloc(N_grid_points, sizeof(float));
    if (electrostat_grid == NULL || vdw_grid_attr == NULL || vdw_grid_rep == NULL) {
        PyErr_SetString(
            PyExc_MemoryError, 
            "Failed to allocate memory for result arrays"
        );
        return NULL;
    }
    // Perform grid generation
    gen_all_grids(
        grid_pos_arr, coords_arr, charges_arr, epsilons_arr, vdw_rs_arr, 
        cc_elec, rad_dielec_const, elec_rep_max, elec_attr_max, 
        vwd_rep_max, vwd_attr_max, N_coords, N_grid_points, 
        use_constant_dielectric, 
        electrostat_grid, vdw_grid_attr, vdw_grid_rep
    );
    // Create NumPy arrays from the result arrays
    npy_intp dims[1] = {N_grid_points};
    PyObject *electrostat_grid_np = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, electrostat_grid
    );
    PyObject *vdw_grid_attr_np = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_grid_attr
    );
    PyObject *vdw_grid_rep_np = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_grid_rep
    );
    if (
        electrostat_grid_np == NULL || vdw_grid_attr_np == NULL || 
        vdw_grid_rep_np == NULL
    ) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result arrays");
        return NULL;
    }
    // Return a tuple of the result arrays
    PyObject *result = PyTuple_Pack(
        3, electrostat_grid_np, vdw_grid_attr_np, vdw_grid_rep_np
    );
    if (result == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result tuple");
        return NULL;
    }
    return result;
}

static PyObject* py_pairwise_dist(PyObject* self, PyObject* args) {
    PyArrayObject *coords, *grid_pos;
    if (!PyArg_ParseTuple(
        args, "O!O!", &PyArray_Type, &coords, &PyArray_Type, &grid_pos
    )) {
        return NULL;
    }
    // Check array dimensions and data type
    if (
        PyArray_TYPE(coords) != NPY_FLOAT32 || 
        PyArray_TYPE(grid_pos) != NPY_FLOAT32
    ) {
        return PyErr_Format(
          PyExc_TypeError, "Expected arrays of float32 for the coord arrays"
        );
    }
    if (PyArray_NDIM(coords) != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected coords array with 2 dimensions");
        return NULL;
    }
    if (PyArray_NDIM(grid_pos) != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected grid_pos array with 1 dimensions");
        return NULL;
    }
    int N_coords = PyArray_DIMS(coords)[0];
    int N_grid_points = PyArray_DIMS(grid_pos)[0];
    // Get NumPy arrays data pointers
    float *coords_arr = (float *)PyArray_DATA(coords);
    float *grid_pos_arr = (float *)PyArray_DATA(grid_pos);
    // Allocate memory for the result array
    float *dists = (float *)malloc(N_coords * N_grid_points * sizeof(float));
    if (dists == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for result array");
        return NULL;
    }
    // Perform pairwise distance calculation
    calc_grid_coord_pairwise_dist(
        grid_pos_arr, coords_arr, N_coords, N_grid_points, dists
    );
    // Create a new NumPy array from the result array
    npy_intp dims[2] = {N_grid_points, N_coords};
    PyObject *result = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, dists);
    if (result == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result array");
        return NULL;
    }
    // Return the result array
    return result;
}

// FFT correlation functions
static PyObject* py_fft_correlate(PyObject* self, PyObject* args) {
    PyArrayObject *recep_grid, *lig_grid;
    int n_threads;
    if (!PyArg_ParseTuple(
      args, "O!O!i", 
      &PyArray_Type, &recep_grid, 
      &PyArray_Type, &lig_grid,
      &n_threads
    )) {
        return NULL;
    }
    // Check array dimensions and data type
    if (PyArray_TYPE(recep_grid) != NPY_FLOAT32 || PyArray_TYPE(lig_grid) != NPY_FLOAT32) {
      PyErr_SetString(PyExc_TypeError, "Expected arrays of float32");
      return NULL;
    }

    if (PyArray_NDIM(recep_grid) != 4) {
      PyErr_SetString(PyExc_TypeError, "Expected receptor grid arrays with 4 dimensions");
      return NULL;
    }

    if (PyArray_NDIM(lig_grid) != 5) {
      PyErr_SetString(PyExc_TypeError, "Expected ligand grid arrays with 5 dimensions");
      return NULL;
    }

    int n_grids = PyArray_DIMS(recep_grid)[0];
    int n_grids_l = PyArray_DIMS(lig_grid)[1];
    if (n_grids != n_grids_l) {
      PyErr_SetString(
        PyExc_TypeError, 
        "Expected same number of grids for both receptor, ligand, and result arrays."
      );
      return NULL;
    }
    // Get array dimensions
    
    int nx = PyArray_DIMS(recep_grid)[1];
    int ny = PyArray_DIMS(recep_grid)[2];
    int nz = PyArray_DIMS(recep_grid)[3];
    size_t N_grid_points = nx * ny * nz;
    // Get lig array dimensions
    int n_orientations = PyArray_DIMS(lig_grid)[0];
    int nx_l = PyArray_DIMS(lig_grid)[2];
    int ny_l = PyArray_DIMS(lig_grid)[3];
    int nz_l = PyArray_DIMS(lig_grid)[4];

    // Get NumPy arrays data pointers
    float *recep_arr = (float *)PyArray_DATA(recep_grid);
    float *lig_arr = (float *)PyArray_DATA(lig_grid);
    
    // Allocate memory for the result array
    size_t n_result_grid_points = N_grid_points * n_orientations;
    float *result_arr = (float *)calloc(
      n_result_grid_points, sizeof(float)
    );
    if (result_arr == NULL) {
      size_t n_megabytes = n_result_grid_points * sizeof(float)/1024/1024;
      return PyErr_Format(
        PyExc_MemoryError, "Failed to allocate %lu MB of memory for the result array",
        n_megabytes
      );
    }

    // Perform FFT correlation
    fft_correlate(
      recep_arr, lig_arr, n_grids, 
      nx, ny, nz, nx_l, ny_l, nz_l, n_orientations,
      n_threads, result_arr
    );
    // Create a new NumPy array from the fft correlation result
    npy_intp dims[4] = {n_orientations, nx, ny, nz};
    PyObject *result = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32, result_arr);
    if (result == NULL) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create result array");
      return NULL;
    }
    // Return the result array
    return result;
}

// Function to pack the top scores and their indices into a tuple to return to Python
static PyObject *pack_top_scores_as_tuple(
  OrienPoseScore *top_scores, const int top_n_scores
  ){
    // Create numpy arrays for scores and indices
    npy_intp dims[1] = {top_n_scores};
    PyObject *out_scores = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyObject *pose_ids = PyArray_SimpleNew(1, dims, NPY_INT);
    PyObject *orientation_ids = PyArray_SimpleNew(1, dims, NPY_INT);

    // Fill the numpy arrays
    for (int i = 0; i < top_n_scores; i++) {
      *(float *)PyArray_GETPTR1((PyArrayObject *)out_scores, i) = top_scores[i].score;
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
    {"pairwise_dist", py_pairwise_dist, METH_VARARGS, "Pairwise distance calculation"},
    {"generate_grids", py_generate_grids, METH_VARARGS, "Generate grids"},
    {"fft_correlate", py_fft_correlate, METH_VARARGS, "FFT Correlation Batch"},
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