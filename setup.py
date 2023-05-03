from setuptools import setup, find_packages

setup(
    name='crimm',
    version='0.0.1',
    install_requires=[
        'biopython>=1.80', # Folks, we need to get past ver 1.79!
        'ipywidgets<=8',
        'nglview>=3.0.3',
        'requests',
    ],
    packages=find_packages(
        where='crimm',
        include=['crimm*']
    ),
)