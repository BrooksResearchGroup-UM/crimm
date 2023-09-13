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

def convert_SMVer_cores(major, minor):
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


class CUDADeviceProp:
    """Class to contain property information about a single CUDA device.
    Should not be instantiated directly, but rather through the CUDAInfo class
    
    Attributes
    ----------
    device_id : int
        The device ID of the device
    name : str
        The name of the device
    cc_major : int
        The major compute capability version of the device
    cc_minor : int
        The minor compute capability version of the device
    compute_capability : str
        The compute capability version of the device
    cores : int
        The number of cores on the device
    threads_per_core : int
        The number of threads per core on the device
    clockrate : float
        The clockrate of the device in MHz
    mem_clockrate : float
        The memory clockrate of the device in MHz
    free_mem : float
        The current amount of free memory on the device in MiB
    total_mem : float
        The total amount of memory on the device in MiB
    """

    # class to contain cudaDeviceProp struct
    # https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
    def __init__(self, device_id: int, cuda_api: ctypes.CDLL) -> None:
        self._device_id = ctypes.c_int()
        self._get_error_str = cuda_api.cuGetErrorString

        if result := cuda_api.cuDeviceGet(
            ctypes.byref(self._device_id), device_id
        ) != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            self._get_error_str(result, ctypes.byref(error_str))
            raise RuntimeError(
                f"Device {self.device_id} cuDeviceGet failed with error code {result}: ",
                f"{error_str.value.decode()}"
            )

        name = b' ' * 100
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        cores = ctypes.c_int()
        threads_per_core = ctypes.c_int()
        clockrate = ctypes.c_int()
        mem_clockrate = ctypes.c_int()
        self._context = ctypes.c_void_p()
        freeMem = ctypes.c_size_t()
        totalMem = ctypes.c_size_t()

        if result := cuda_api.cuDeviceGetName(
            ctypes.c_char_p(name), len(name), self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuDeviceGetName')
            self.name = None
        else:
            name = name.decode()
            self.name = name.rstrip().rstrip('\x00')

        if result := cuda_api.cuDeviceComputeCapability(
            ctypes.byref(cc_major), ctypes.byref(cc_minor), self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuDeviceComputeCapability')
            self.cc_major, self.cc_minor = None, None
        else:
            self.cc_major, self.cc_minor = cc_major.value, cc_minor.value

        if result := cuda_api.cuDeviceGetAttribute(
            ctypes.byref(cores),
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuDeviceGetAttribute')
            self.cores = None
            self.cuda_cores = None
        else:
            self.cores = cores.value
            self.cuda_cores = self.cores * convert_SMVer_cores(self.cc_major, self.cc_minor) or "unknown"
            if result := cuda_api.cuDeviceGetAttribute(
                ctypes.byref(threads_per_core),
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 
                self._device_id
            ) != CUDA_SUCCESS:
                self._reveal_error_code(result, 'cuDeviceGetAttribute')
                self.threads_per_core = None
                self.concurrent_threads = None
            else:
                self.threads_per_core = threads_per_core.value
                self.concurrent_threads = self.cores * self.threads_per_core
                
        if result := cuda_api.cuDeviceGetAttribute(
            ctypes.byref(clockrate),
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuDeviceGetAttribute')
            self.clockrate = None
        else:
            self.clockrate = clockrate.value / 1000. #MHz
        
        if result := cuda_api.cuDeviceGetAttribute(
            ctypes.byref(mem_clockrate),
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, 
            self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuDeviceGetAttribute')
            self.mem_clockrate = None
        else:
            self.mem_clockrate = mem_clockrate.value / 1000. #MHz
        
        if hasattr(cuda_api, 'cuCtxCreate_v2'):
            create_context = cuda_api.cuCtxCreate_v2
            self._get_mem_info = cuda_api.cuMemGetInfo_v2
        else:
            create_context = cuda_api.cuCtxCreate
            self._get_mem_info = cuda_api.cuMemGetInfo
        self._destroy_context = cuda_api.cuCtxDetach
        
        if result := create_context(
            ctypes.byref(self._context), 0, self._device_id
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuCtxCreate')
            self._free_mem = None
            self.total_mem = None
        elif result := self._get_mem_info(
            ctypes.byref(freeMem), ctypes.byref(totalMem)
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuMemGetInfo')
            self._free_mem = None
            self.total_mem = None
        else:
            self._free_mem = freeMem.value / 1024**2 #MB
            self.total_mem = totalMem.value / 1024**2 #MB
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.detach()

    def detach(self):
        """Free the memory for CUDA context and destroy handle"""
        if self._context is not None:
            self._destroy_context(self._context)
            self._context = None

    def _reveal_error_code(self, result: int, func_name: str) -> None:
        error_str = ctypes.c_char_p()
        self._get_error_str(result, ctypes.byref(error_str))
        warnings.warn(
            f"Device {self.device_id} {func_name} failed with error code {result}: "
            f"{error_str.value.decode()}"
        )

    def _update_mem(self):
        if self._context is None:
            self._free_mem = None
            return

        free_mem = ctypes.c_size_t()
        total_mem = ctypes.c_size_t()

        if result := self._get_mem_info(
            ctypes.byref(free_mem),
            ctypes.byref(total_mem)
        ) != CUDA_SUCCESS:
            self._reveal_error_code(result, 'cuMemGetInfo')
            self._free_mem = None
            return
        self._free_mem = free_mem.value / 1024**2 #MB

    @property
    def free_mem(self):
        """Returns the free memory on the device in MiB."""
        self._update_mem()
        return self._free_mem

    @property
    def device_id(self):
        return self._device_id.value

    @property
    def compute_capability(self):
        """Returns the compute capability of the device."""
        return '.'.join([str(self.cc_major), str(self.cc_minor)])

    @property
    def is_detached(self):
        return self._context is None

    def __str__(self) -> str:
        info_str = (
            f"<Device {self.device_id}: {self.name} "
            f"totalMem={self.total_mem} MiB"
        )
        if self.free_mem is not None:
            info_str += f", freeMem={self.free_mem} MiB"
        info_str += ">"
        return info_str
    
    def __repr__(self) -> str:
        return str(self)
    
    def report(self) -> None:
        """Prints a report of the device's information."""
        print(f"<Device {self.device_id}: {self.name}>")
        print(f"\tCompute Capability: {self.compute_capability}")
        print(f"\tCUDA Cores: {self.cuda_cores}")
        print(f"\tClockrate: {self.clockrate} MHz")
        print(f"\tMemory Clockrate: {self.mem_clockrate} MHz")
        print(f"\tConcurrent Threads: {self.concurrent_threads}")
        print(f"\tTotal Memory: {self.total_mem} MiB")
        if self.free_mem is not None:
            print(f"\tFree Memory: {self.free_mem} MiB")


class CUDAInfo:
    """Container for available CUDA device information.
    Note, the CUDA driver API must be installed for this to work. The class will 
    attempt to load the CUDA driver API library from the default locations for 
    the current platform. If the library cannot be found, OSError will be 
    raised. If the library is found, but the CUDA driver API cannot be loaded
    from it, RuntimeError will be raised. 
    While using this utility for free memory monitoring, the class will allocate
    about 220 MiB of memory on each device for memory monitoring. This memory 
    will be deallocated when the context is detached. Thus, it is recommended to 
    use a context manager; the memory will be automatically freed when the 
    context is exited.

    Attributes:
        devices (list): List of CUDA devices available on the system.
        n_gpus (int): Number of CUDA devices available on the system.

    Usage:
    >>> from crimm.Utils.cuda_info import CUDAInfo
    ## with context manager
    >>> with CUDAInfo() as cuda_info:
    >>>     cuda_info.report()
    <CUDA Device Info for 1 GPU(s)>
    <Device 0: NVIDIA GeForce GTX 1080 Ti>
        Compute Capability: 6.1
        CUDA Cores: 3584
        Clockrate: 1582.0 MHz
        Memory Clockrate: 5505.0 MHz
        Concurrent Threads: 57344
        Total Memory: 11172.1875 MiB
        Free Memory: 11032.125 MiB
    ## or without context manager
    >>> cuda_info = CUDAInfo()
    >>> cuda_info.devices
    [<CUDA Device 0: NVIDIA GeForce GTX 1080 Ti>]
    >>> cuda_info.n_gpus
    1
    >>> cuda_info.devices[0].name # or cuda_info[0].name
    'NVIDIA GeForce GTX 1080 Ti'
    >>> cuda_info.devices[0].compute_capability
    '6.1'
    >>> cuda_info.devices[0].free_mem #MiB of free memory
    11032.125
    >>> cuda_info.detach() #deallocate CUDA memory and destroy CUDA context on all devices
    >>> cuda_info.devices[0].free_mem
    None
    >>> cuda_info.devices[0].is_detached
    True
    """

    # possible CUDA driver API library names
    # for linux, it's usually located in /usr/lib64/libcuda.so 
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    def __init__(self) -> None:
        self.devices = []
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

        self.n_gpus = self._get_num_gpus()
        self._enumerate_devices()

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
                "Could not find CUDA driver. "
                "Failed to load any of: " + ' '.join(self.libnames)
            )

    def _get_num_gpus(self):
        nGpus = ctypes.c_int()
        result = ctypes.c_int()
        result = self.cuda_api.cuDeviceGetCount(ctypes.byref(nGpus))
        if result == CUDA_SUCCESS:
            return nGpus.value
        error_str = ctypes.c_char_p()
        self.cuda_api.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(
            "cuDeviceGetCount failed with error code "
            f"{result}: {error_str.value.decode()}"
        )

    def _enumerate_devices(self):
        for i in range(self.n_gpus):
            device = CUDADeviceProp(i, self.cuda_api)
            self.devices.append(device)

    def __str__(self) -> str:
        return f"<CUDAInfo: {self.n_gpus} GPUs>"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for device in self.devices:
            device.__exit__(exc_type, exc_value, traceback)

    def __del__(self):
        for device in self.devices:
            device.detach()

    def __getitem__(self, index):
        return self.devices[index]
    
    def __iter__(self):
        return iter(self.devices)
    
    def __len__(self):
        return len(self.devices)
    
    def report(self):
        print(f"<CUDA Device Info for {self.n_gpus} GPU(s)>")
        for device in self.devices:
            device.report()
    
    def detach(self):
        for device in self.devices:
            device.detach()

def report_cuda_devices():
    """Prints a report of the CUDA devices on the system."""
    with CUDAInfo() as cuda_info:
        cuda_info.report()
    return 0

if __name__=="__main__":
    sys.exit(report_cuda_devices())