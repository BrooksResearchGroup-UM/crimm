from setuptools import setup, Extension
import numpy

# Define the extension module
fft_correlate_module = Extension('fft_correlate', sources=['fft_correlate.c'],
                                 include_dirs=[numpy.get_include(), '/home/truman/.conda/envs/pcm-devel/include/'],
                                 libraries=['fftw3f', 'fftw3f_threads', 'm', 'gomp'],
                                 library_dirs=['/home/truman/.conda/envs/pcm-devel/include/'],
                                 define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                                 extra_compile_args=[
                                    '-Ofast', 
                                    # '-pthread',
                                    '-fopenmp'
                                    ],
                                # extra_compile_args=[
                                #     '-O0', '-g', 
                                #     # '-pthread',
                                #     '-fopenmp']
                                 )

# Run the setup
setup(ext_modules=[fft_correlate_module])
