from setuptools import setup, Extension
import numpy

# Define the extension module
fft_correlate_module = Extension(
    'crimm.fft_docking', 
    sources=[
        'src/fft_docking/py_bindings.c',
        'src/fft_docking/fft_correlate.c',
        'src/fft_docking/rank_poses.c',
        'src/fft_docking/grid_gen.c',
        'src/fft_docking/lig_grid_gen.c'
    ],
    include_dirs=[
        numpy.get_include(),
        '/home/truman/.conda/envs/pcm-devel/include/',
        './'
    ],
    libraries=['fftw3f', 'fftw3f_threads', 'm', 'gomp'],
    library_dirs=[
        '/home/truman/.conda/envs/pcm-devel/include/',
    ],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    extra_compile_args=[
    '-Ofast', 
    '-g',
    '-fopenmp'
    ]
)

setup(
    ext_modules=[fft_correlate_module]
)