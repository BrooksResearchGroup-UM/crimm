#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "rank_poses.h"

// Comparator function for max heap
int comparator(const void *p, const void *q) {
    float l = ((ScoreIndexPair *)p)->score;
    float r = ((ScoreIndexPair *)q)->score;
    return (l < r) - (l > r);
}

int comparator_3(const void *p, const void *q) {
    float l = ((OrienPoseScore *)p)->score;
    float r = ((OrienPoseScore *)q)->score;
    return (l < r) - (l > r);
}

// Function to get top n scores and their indices (max heap)
void get_top_n_scores(
  ScoreIndexPair* scores, int n, int top_n_poses, ScoreIndexPair* top_scores
) {
    // Initialize max heap
    ScoreIndexPair* heap = (ScoreIndexPair*) malloc(
      top_n_poses * sizeof(ScoreIndexPair)
    );

    for (int i = 0; i < top_n_poses; i++) {
        heap[i] = scores[i];
      }
    qsort(heap, top_n_poses, sizeof(ScoreIndexPair), comparator);

    // Find the top n scores and their indices
    for (int i = top_n_poses; i < n; i++) {
      if (scores[i].score < heap[0].score) {
          // Replace the root of the heap
          heap[0] = scores[i];
          // Re-sort the heap
          qsort(heap, top_n_poses, sizeof(ScoreIndexPair), comparator);
      }
    }

    // Copy the heap to the output array
    for (int i = 0; i < top_n_poses; i++) {
      top_scores[top_n_poses-i-1].index = heap[i].index;
      top_scores[top_n_poses-i-1].score = heap[i].score;
    }

    free(heap);
}

void combine_top_scores(
  ScoreIndexPair *scores_by_orientation, int n_orientations, int n_scores,
  int n_top_scores_per_orientation, int n_top_scores, OrienPoseScore *top_scores
){
  int cur_idx;
  // Initialize max heap
  OrienPoseScore* heap = (OrienPoseScore*) malloc(
    n_top_scores * sizeof(OrienPoseScore)
  );
  // Fill the heap with the first n_top_scores scores
  for (int i = 0; i < n_orientations; i++) {
    if (i * n_top_scores_per_orientation >= n_top_scores) {
      break;
    }
    for (int j = 0; j < n_top_scores_per_orientation; j++) {
      cur_idx = i * n_top_scores_per_orientation + j;
      heap[cur_idx].pose_id = scores_by_orientation[cur_idx].index;
      heap[cur_idx].score = scores_by_orientation[cur_idx].score;
      heap[cur_idx].orientation_id = i;
      if (cur_idx == n_top_scores - 1) {
        // Build the heap when it's full
        qsort(heap, n_top_scores, sizeof(OrienPoseScore), comparator_3);
        break;
      }
    }
  }
  // Find the top n scores and their indices
  for (int i = 0; i < n_orientations; i++) {
    for (int j = 0; j < n_top_scores_per_orientation; j++) {
      cur_idx = i * n_top_scores_per_orientation + j;
      // If the current score is greater than the root of the heap, skip
      // the rest of the scores for this orientation. Since the scores
      // are sorted, the rest of the scores will also be greater than the root
      if (scores_by_orientation[cur_idx].score > heap[0].score) {
        break;
      }
      // Replace the root of the heap
      heap[0].pose_id = scores_by_orientation[cur_idx].index;
      heap[0].score = scores_by_orientation[cur_idx].score;
      heap[0].orientation_id = i;
      // Re-sort the heap
      qsort(heap, n_top_scores, sizeof(OrienPoseScore), comparator_3);
    }
  }
  // Copy the heap to the output array reverse the order for a min heap
  for (int i = 0; i < n_top_scores; i++) {
    top_scores[n_top_scores-i-1].pose_id = heap[i].pose_id;
    top_scores[n_top_scores-i-1].orientation_id = heap[i].orientation_id;
    top_scores[n_top_scores-i-1].score = heap[i].score;
  }
  free(heap);
}

// Function to find negative values in an array and store their indices
// Returns the number of negative values
int find_neg_vals(const float *arr, const int n, int *neg_val_ids) {
    int neg_val_counter = 0; 
    // The index 0 is temp place holder for the index of positive values
    // neg val idx start at 1 to avoid overwriting the first index
    int cur_id = 0;
    for (int i = 0; i < n; i++) {
      // add tolarence of 0.0001 to avoid -0.0
      bool is_neg = signbit(arr[i]+0.0001); // 1 if negative, 0 if positive
      neg_val_counter += is_neg; // Increment counter if negative
      cur_id = is_neg * neg_val_counter; // 0 if positive, cur_id > 0 if negative
      neg_val_ids[cur_id] = i;
    }
    return neg_val_counter;
}

// Function to rank negative values in an array (ascending order)
void sort_neg_vals(
  const float *scores, const int n_scores, const int n_top_scores, 
  ScoreIndexPair *top_scores
) {
  int *neg_val_ids = (int *)malloc((n_scores+1) * sizeof(int));
  int n_neg_scores = find_neg_vals(scores, n_scores, neg_val_ids);
  ScoreIndexPair *neg_scores = (ScoreIndexPair *)malloc(
    n_neg_scores * sizeof(ScoreIndexPair)
  );
  for (int i = 0; i < n_neg_scores; i++) {
    // We use i+1 b/c n_neg_scores starts at 1
    neg_scores[i].score = scores[neg_val_ids[i+1]];
    neg_scores[i].index = neg_val_ids[i+1];
  }
  if (n_neg_scores < n_top_scores) {
    get_top_n_scores(neg_scores, n_neg_scores, n_neg_scores, top_scores);
  }
  else {
    get_top_n_scores(neg_scores, n_neg_scores, n_top_scores, top_scores);
  }
  free(neg_val_ids);
  free(neg_scores);
}

void rank_poses(
  float *scores, const int n_orientations, const int n_scores,
  const int sample_factor, const int n_top_scores, int n_threads,
  OrienPoseScore *top_scores
) {
  int n_top_score_per_orientation = n_top_scores / sample_factor;
  // Allocate memory for the top scores for each orientation
  ScoreIndexPair *top_scores_by_orientation = (ScoreIndexPair *)malloc(
    n_orientations * n_top_score_per_orientation * sizeof(ScoreIndexPair)
  );
  memset(
    top_scores_by_orientation, 0, 
    n_orientations * n_top_score_per_orientation * sizeof(ScoreIndexPair)
  );
  // Sort scores for each orientation
  #pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < n_orientations; i++) {
    float *cur_scores = scores + i * n_scores;
    ScoreIndexPair *cur_top_scores = (
      top_scores_by_orientation + i * n_top_score_per_orientation
    );
    sort_neg_vals(
      cur_scores, n_scores, n_top_score_per_orientation, cur_top_scores
    );
  }
  // Sort top score for all orientations combined
  combine_top_scores(
    top_scores_by_orientation, n_orientations, n_scores,
    n_top_score_per_orientation, n_top_scores, top_scores
  );

  free(top_scores_by_orientation);
}

// Function to pack the top scores and their indices into a tuple to return to Python
PyObject *pack_top_scores_as_tuple(
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

PyObject* py_rank_poses(PyObject* self, PyObject* args) {
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