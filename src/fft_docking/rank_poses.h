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

// Function to rank poses based on docking scores
void rank_poses(
  float *scores, const size_t n_orientations, const size_t n_scores,
  const size_t sample_factor, const size_t n_top_scores, int n_threads,
  OrienPoseScore *top_scores
);

// Function to calculate pairwise RMSD between conformations
void calc_pairwise_rmsd(
  const float *conf_coords, const size_t n_confs, const size_t n_atoms,
  float *rmsd
);

#endif