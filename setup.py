from setuptools import setup, find_packages

setup(
    name='crimm',
    version='0.0.1',
    install_requires=[
        'biopython',
        'nglview',
    ],
    packages=find_packages(
        # All keyword arguments below are optional:
        where='crimm',
        include=['crimm*']
    ),
)