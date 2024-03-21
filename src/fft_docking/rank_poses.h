#ifndef RANK_POSES_H
#define RANK_POSES_H


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
  float *scores, const int n_orientations, const int n_scores,
  const int sample_factor, const int n_top_scores, int n_threads,
  OrienPoseScore *top_scores
);

#endif