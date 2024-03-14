#ifndef RANK_POSES_H
#define RANK_POSES_H

#include <Python.h>
typedef struct {
    int index;
    float score;
} ScoreIndexPair;

typedef struct {
    int orientation_id;
    int pose_id;
    float score;
} OrienPoseScore;

void rank_poses(
    int num_poses, int num_orientations, float *scores, int *pose_ids, 
    int *orientation_ids, ScoreIndexPair *sorted_poses, 
    OrienPoseScore *sorted_orientations
);

PyObject* py_rank_poses(PyObject* self, PyObject* args);

#endif