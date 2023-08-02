# Description: Check if CUDA is available on the system
# -*- coding: utf-8 -*-

"""
Outputs some information on CUDA-enabled devices on your computer,
including current memory usage.
It's a port of https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
from C to Python with ctypes, so it can run without compiling anything. Note
that this is a direct translation with no attempt to make the code Pythonic.
It's meant as a general demonstration on how to obtain CUDA device information
from Python without resorting to nvidia-smi or a compiled Python extension.
Author: Jan Schlüter, Ziqiao "Truman" Xu (徐梓乔)
License: MIT (https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549#gistcomment-3870498)
"""

import sys
import ctypes
import warnings
from dataclasses import dataclass

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
# Table to convert CUDA architectural version number into core count per SM,
# data from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
CUDA_CORE_TABLE = {
    # (major, minor): cores
    (1, 0): 8,    # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,   # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,   # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,   # Volta
    (7, 2): 64,   # Xavier
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,   # Ada
    (9, 0): 128,   # Hopper
}

def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return CUDA_CORE_TABLE.get((major, minor), 0)

def is_cuda_available():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return False

    result = ctypes.c_int()
    result = cuda.cuInit(0)
    if result == CUDA_SUCCESS:
        return True

    return False

@dataclass
class CUDADeviceProp:
    # class to contain cudaDeviceProp struct
    # https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    device_id: int
    name: str
    cc_major: int
    cc_minor: int
    cores: int
    threads_per_core: int
    clockrate: int
    freeMem: int
    totalMem: int

    def __str__(self) -> str:
        return f"Device {self.device_id}: {self.name}"
    
class CUDAInfo:
    # possible CUDA driver API library names
    # for linux, it's usually located in /usr/lib64/libcuda.so 
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    def __init__(self) -> None:
        self.cuda_api = self._find_cuda_api_lib()
        init_result = ctypes.c_int()
        init_result = self.cuda_api.cuInit(0)
        if init_result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            self.cuda_api.cuGetErrorString(init_result, ctypes.byref(error_str))
            raise OSError(
                "cuInit failed with error code "
                f"{init_result}: {error_str.value.decode()}"
            )

        self._nGpus = ctypes.c_int()
        self.devices = []

    @property
    def nGpus(self):
        return self._nGpus.value
    
    def _find_cuda_api_lib(self):
        for libname in self.libnames:
            try:
                return ctypes.CDLL(libname)
            except OSError:
                continue
            else:
                break
        else:
            raise OSError(
                "Could not find CUDA driver."
                "Failed to load any of: " + ' '.join(self.libnames)
            )

    def _get_num_gpus(self):
        result = ctypes.c_int()
        result = self.cuda_api.cuDeviceGetCount(ctypes.byref(self._nGpus))
        if result == CUDA_SUCCESS:
            return
        error_str = ctypes.c_char_p()
        self.cuda_api.cuGetErrorString(result, ctypes.byref(error_str))
        raise OSError(
            "cuDeviceGetCount failed with error code "
            f"{result}: {error_str.value.decode()}"
        )

    def _populate_device_prop(self):
        result = ctypes.c_int()
        for i in range(self.nGpus):
            result = self.cuda_api.cuDeviceGet(ctypes.byref(device), i)
            if result != CUDA_SUCCESS:
                self.cuda_api.cuGetErrorString(result, ctypes.byref(error_str))
                print("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
        
def report_cuda_devices():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))
    
    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()
    result = ctypes.c_int()
    result = cuda.cuInit(0)

    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    print("Found %d device(s)." % nGpus.value)
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
            return 1
        print("Device: %d" % i)
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            print("  Name: %s" % (name.split(b'\0', 1)[0].decode()))
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            print("  Compute Capability: %d.%d" % (cc_major.value, cc_minor.value))
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            print("  Multiprocessors: %d" % cores.value)
            print("  CUDA Cores: %s" % (cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value) or "unknown"))
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                print("  Concurrent threads: %d" % (cores.value * threads_per_core.value))
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            print("  GPU clock: %g MHz" % (clockrate.value / 1000.))
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
            print("  Memory clock: %g MHz" % (clockrate.value / 1000.))
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                print("  Total Memory: %ld MiB" % (totalMem.value / 1024**2))
                print("  Free Memory: %ld MiB" % (freeMem.value / 1024**2))
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                print("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
            cuda.cuCtxDetach(context)
    return 0

if __name__=="__main__":
    sys.exit(report_cuda_devices())