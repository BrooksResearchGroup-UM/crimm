#ifndef FFT_CORRELATE_H
#define FFT_CORRELATE_H
#include <Python.h>
#include <numpy/arrayobject.h>

void fft_correlate_batch(
  PyArrayObject *recep_grid, PyArrayObject *result, int n_threads
);

void sum_grids(
  PyArrayObject *grids, PyArrayObject *result, int roll_steps
);

#endif