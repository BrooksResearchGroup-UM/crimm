#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include "fft_correlate.h"
#include "rank_poses.h"
#include "grid_gen.h"

// TODO: Add check for NPY_ARRAY_C_CONTIGUOUS flag
// Wrapper function to be called from Python
// Generate Protein Grids
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
    PyObject *npy_electrostat_grid = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, electrostat_grid
    );
    PyObject *npy_vdw_grid_attr = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_grid_attr
    );
    PyObject *npy_vdw_grid_rep = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_grid_rep
    );
    if (
        npy_electrostat_grid == NULL || npy_vdw_grid_attr == NULL || 
        npy_vdw_grid_rep == NULL
    ) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result arrays");
        return NULL;
    }
    // Enable ownership of the result arrays by the NumPy arrays
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_electrostat_grid, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_grid_attr, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_grid_rep, NPY_ARRAY_OWNDATA);

    // Return a tuple of the result arrays
    PyObject *py_return = PyTuple_Pack(
        3, npy_electrostat_grid, npy_vdw_grid_attr, npy_vdw_grid_rep
    );
    if (py_return == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result tuple");
        return NULL;
    }
    return py_return;
}

static PyObject* py_pairwise_dist(PyObject* self, PyObject* args) {
    PyArrayObject *grid_pos, *coords;
    if (!PyArg_ParseTuple(
        args, "O!O!", &PyArray_Type, &grid_pos, &PyArray_Type, &coords
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
    if (PyArray_NDIM(grid_pos) != 2) {
        PyErr_SetString(PyExc_TypeError, "Expected grid_pos array with 2 dimensions");
        return NULL;
    }
    int N_coords = PyArray_DIMS(coords)[0];
    int N_grid_points = PyArray_DIMS(grid_pos)[0];
    // Get NumPy arrays data pointers
    float *grid_pos_arr = (float *)PyArray_DATA(grid_pos);
    float *coords_arr = (float *)PyArray_DATA(coords);
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
    PyObject *npy_dists = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, dists);
    // Enable ownership of the result arrays by the NumPy arrays
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_dists, NPY_ARRAY_OWNDATA);

    if (npy_dists == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create result array");
        return NULL;
    }
    // Return the result array
    return npy_dists;
}

// Generate ligand grids
static PyObject* py_rotate_gen_lig_grids(PyObject* self, PyObject* args){
    float grid_spacing;
    PyArrayObject *charges, *vdw_attr_factors, *vdw_rep_factors;
    PyArrayObject *coords, *quats;
    
    if (!PyArg_ParseTuple(args, "fO!O!O!O!O!", 
        &grid_spacing,
        &PyArray_Type, &charges, &PyArray_Type, &vdw_attr_factors, 
        &PyArray_Type, &vdw_rep_factors, 
        &PyArray_Type, &coords, &PyArray_Type, &quats
    )){
        return NULL;
    }
    int N_coords = PyArray_DIM(coords, 0);
    int N_quats = PyArray_DIM(quats, 0);
    float *charges_data = PyArray_DATA(charges);
    float *vdw_attr_factors_data = PyArray_DATA(vdw_attr_factors);
    float *vdw_rep_factors_data = PyArray_DATA(vdw_rep_factors);
    float *coords_data = PyArray_DATA(coords);
    float *quats_data = PyArray_DATA(quats);
    float *rot_coords = malloc(N_quats * N_coords * 3 * sizeof(float));
    float max_dist = get_max_pairwise_dist(coords_data, N_coords);
    // +1 account for the last edge of the grid, e,g. a cube of 3x3x3 has 4 
    // grid points per edge 
    int grid_dim = (int)ceil(max_dist / grid_spacing) + 1;
    // we use cube grid to promote throughput
    size_t N_grid_points = grid_dim * grid_dim * grid_dim; 
    float *elec_grids = calloc(N_grid_points * N_quats, sizeof(float));
    float *vdw_grids_attr = calloc(N_grid_points * N_quats, sizeof(float));
    float *vdw_grids_rep = calloc(N_grid_points * N_quats, sizeof(float));

    rotate_gen_lig_grids(
        grid_spacing, charges_data, vdw_attr_factors_data, vdw_rep_factors_data, 
        coords_data, N_coords, quats_data, N_quats, grid_dim, 
        rot_coords, elec_grids, vdw_grids_attr, vdw_grids_rep
    );
    
    npy_intp rot_coords_dims[3] = {N_quats, N_coords, 3};
    npy_intp grid_dims[4] = {N_quats, grid_dim, grid_dim, grid_dim};
    PyObject *npy_rot_coords = PyArray_SimpleNewFromData(
        3, rot_coords_dims, NPY_FLOAT32, rot_coords
    );
    PyObject *npy_elec_grids = PyArray_SimpleNewFromData(
        4, grid_dims, NPY_FLOAT32, elec_grids
    );
    PyObject *npy_vdw_grids_attr = PyArray_SimpleNewFromData(
        4, grid_dims, NPY_FLOAT32, vdw_grids_attr
    );
    PyObject *npy_vdw_grids_rep = PyArray_SimpleNewFromData(
        4, grid_dims, NPY_FLOAT32, vdw_grids_rep
    );
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_rot_coords, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_elec_grids, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_grids_attr, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_grids_rep, NPY_ARRAY_OWNDATA);
    return Py_BuildValue(
        "OOOO", 
        npy_rot_coords, npy_elec_grids, npy_vdw_grids_attr, npy_vdw_grids_rep
    );
}

// Calculate VDW energy factors
static PyObject* py_calc_vdw_energy_factors(PyObject* self, PyObject* args){
    PyArrayObject *epsilons, *vdw_rs;
    int N_coords;
    if (!PyArg_ParseTuple(args, "O!O!", 
        &PyArray_Type, &epsilons, &PyArray_Type, &vdw_rs
    )){
        return NULL;
    }
    N_coords = PyArray_DIM(epsilons, 0);
    float *epsilons_data = PyArray_DATA(epsilons);
    float *vdw_rs_data = PyArray_DATA(vdw_rs);
    float *vdw_attr_factors = malloc(N_coords * sizeof(float));
    float *vdw_rep_factors = malloc(N_coords * sizeof(float));
    calc_vdw_energy_factors(
        epsilons_data, vdw_rs_data, N_coords, vdw_attr_factors, vdw_rep_factors
    );
    npy_intp dims[1] = {N_coords};
    PyObject* npy_vdw_attr_factors = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_attr_factors
    );
    PyObject* npy_vdw_rep_factors = PyArray_SimpleNewFromData(
        1, dims, NPY_FLOAT32, vdw_rep_factors
    );
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_attr_factors, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_vdw_rep_factors, NPY_ARRAY_OWNDATA);
    return Py_BuildValue("OO", npy_vdw_attr_factors, npy_vdw_rep_factors);
}

// Batch quaternion rotation on coordinates
static PyObject* py_batch_quatornion_rotate(PyObject* self, PyObject* args){
    PyArrayObject *coords, *quats;
    int N_coords, N_quats;
    if (!PyArg_ParseTuple(args, "O!O!", 
        &PyArray_Type, &coords, &PyArray_Type, &quats
    )){
        return NULL;
    }
    N_coords = PyArray_DIM(coords, 0);
    N_quats = PyArray_DIM(quats, 0);
    float* coords_data = PyArray_DATA(coords);
    float* quats_data = PyArray_DATA(quats);
    float* rot_coords = malloc(N_quats * N_coords * 3 * sizeof(float));
    batch_quaternion_rotate(
        coords_data, N_coords, quats_data, N_quats, rot_coords
    );
    npy_intp rot_coords_dims[3] = {N_quats, N_coords, 3};
    PyObject* npy_rot_coords = PyArray_SimpleNewFromData(
        3, rot_coords_dims, NPY_FLOAT32, rot_coords
    );
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_rot_coords, NPY_ARRAY_OWNDATA);
    return npy_rot_coords;
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
      return PyErr_Format(
        PyExc_TypeError, 
        "Expected same number of grids for both receptor, ligand, and result arrays. "
        "Receptor grids: %d, Ligand grids: %d", n_grids, n_grids_l
      );
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
    size_t n_result_grid_points = N_grid_points * n_orientations * n_grids;
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
    npy_intp dims[5] = {n_orientations, n_grids, nx, ny, nz};
    PyObject *npy_result = PyArray_SimpleNewFromData(5, dims, NPY_FLOAT32, result_arr);
    if (npy_result == NULL) {
      PyErr_SetString(PyExc_MemoryError, "Failed to create result array");
      return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_result, NPY_ARRAY_OWNDATA);
    // Return the result array
    return npy_result;
}

// Function to sum grids
static PyObject *py_sum_grids(PyObject *self, PyObject *args){
        PyArrayObject *grids;
        int n_grids;
        size_t n_orientations, n_scores;
        if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &grids)) {
            return NULL;
        }
        // Check array dimensions and data type
        if (PyArray_TYPE(grids) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_TypeError, "Expected grids array of float32");
            return NULL;
        }
        int n_dims = PyArray_NDIM(grids);
        npy_intp out_dims[n_dims-1];
        // Check array dimensions and data type
        if (n_dims == 5) {
            // Get array dimensions
            n_orientations = PyArray_DIMS(grids)[0];
            n_grids = PyArray_DIMS(grids)[1];
            int nx = PyArray_DIMS(grids)[2];
            int ny = PyArray_DIMS(grids)[3];
            int nz = PyArray_DIMS(grids)[4];
            n_scores = nx * ny * nz;
            out_dims[0] = n_orientations;
            out_dims[1] = nx;
            out_dims[2] = ny;
            out_dims[3] = nz;
        } else if (n_dims == 3) {
            // Get array dimensions
            n_orientations = PyArray_DIMS(grids)[0];
            n_grids = PyArray_DIMS(grids)[1];
            n_scores = PyArray_DIMS(grids)[2];
            out_dims[0] = n_orientations;
            out_dims[1] = n_scores;
        } else {
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected scores array of float32 with 5 dimension \
            (n_grids, n_orientations, nx, ny, nz), or 2 dimension \
            (n_grids, n_orientations, n_scores)."
        );
        return NULL;
        }
        
        // Get NumPy arrays data pointers
        float *grids_arr = (float *)PyArray_DATA(grids);
        // Allocate memory for the result array
        size_t n_grid_points = n_scores * n_orientations;
        float *result_arr = (float *)calloc(n_grid_points, sizeof(float));
        if (result_arr == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for result array");
            return NULL;
        }
        // Perform grid summation
        sum_grids(grids_arr, result_arr, n_orientations, n_grids, n_scores);
        // Create a new NumPy array from the result array
        PyObject *npy_result = PyArray_SimpleNewFromData(
            n_dims-1, out_dims, NPY_FLOAT32, result_arr
        );
        if (npy_result == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create result array");
            return NULL;
        }
        PyArray_ENABLEFLAGS((PyArrayObject*) npy_result, NPY_ARRAY_OWNDATA);
        // Return the result array
        return npy_result;
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

static PyObject* py_rank_poses(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject *scores;
    int top_n_poses, sample_factor, n_threads;
    size_t n_orientations, n_scores;
    static char* keywords[] = {
        "", "top_n_poses", "sample_factor", "n_threads", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "O!kki", keywords,
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
    float *scores_arr = (float *)PyArray_DATA(scores);
    rank_poses(
      scores_arr, n_orientations, n_scores, sample_factor, 
      top_n_poses, n_threads, top_scores
    );
    PyObject *result = pack_top_scores_as_tuple(top_scores, top_n_poses);
    return result;
}

static PyObject* py_calc_pairwise_rmsd(PyObject* self, PyObject* args){
    PyArrayObject *conf_coords;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &conf_coords)){
        return NULL;
    }
    // Check array dimensions and data type
    if (PyArray_TYPE(conf_coords) != NPY_FLOAT32){
        PyErr_SetString(PyExc_TypeError, "Expected conf_coords array of float32.");
        return NULL;
    }
    if (PyArray_NDIM(conf_coords) != 3){
        PyErr_SetString(PyExc_TypeError, "Expected conf_coords array with 3 dimensions.");
        return NULL;
    }
    int N_confs = PyArray_DIM(conf_coords, 0);
    if (N_confs < 2) {
        PyErr_SetString(PyExc_TypeError, "Expected at least 2 conformations.");
        return NULL;
    }
    int N_atoms = PyArray_DIM(conf_coords, 1);
    if (PyArray_DIM(conf_coords, 2) != 3){
        PyErr_SetString(
            PyExc_TypeError, 
            "Expected conf_coords array with 3 columns (x, y, z) for coordinates."
        );
        return NULL;
    }
    float *conf_coords_data = PyArray_DATA(conf_coords);
    size_t N_pairs = (N_confs-1)*N_confs/2;
    float *rmsd = malloc(N_pairs * sizeof(float));
    calc_pairwise_rmsd(conf_coords_data, N_confs, N_atoms, rmsd);
    npy_intp dims[1] = {N_pairs};
    PyObject *npy_rmsd = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, rmsd);
    PyArray_ENABLEFLAGS((PyArrayObject*) npy_rmsd, NPY_ARRAY_OWNDATA);
    return npy_rmsd;
}

// Method table
static PyMethodDef FftDockingMethods[] = {
    {"pairwise_dist", py_pairwise_dist, METH_VARARGS, "Pairwise distance calculation"},
    {"generate_grids", py_generate_grids, METH_VARARGS, "Generate grids"},
    {"calc_vdw_energy_factors", py_calc_vdw_energy_factors, METH_VARARGS, "Calculate VDW energy factors"},
    {"rotate_gen_lig_grids", py_rotate_gen_lig_grids, METH_VARARGS, "Rotate and generate ligand grids"},
    {"batch_quaternion_rotate", py_batch_quatornion_rotate, METH_VARARGS, "Batch quaternion rotation"},
    {"fft_correlate", py_fft_correlate, METH_VARARGS, "FFT Correlation Batch"},
    {"sum_grids", py_sum_grids, METH_VARARGS, "Sum grids"},
    {"rank_poses", (PyCFunction)py_rank_poses, METH_VARARGS | METH_KEYWORDS, "Rank poses"},
    {"calc_pairwise_rmsd", py_calc_pairwise_rmsd, METH_VARARGS, "Pairwise RMSD calculation"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef fftdockingmodule = {
    PyModuleDef_HEAD_INIT,
    "fft_docking",   // name of module
    NULL,              // module documentation, may be NULL
    -1,                // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    FftDockingMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_fft_docking(void) {
    import_array();
    return PyModule_Create(&fftdockingmodule);
}