# coding=utf-8
# from distutils.core import setup, Extension
from setuptools import find_packages, setup, Extension
from glob import glob
from distutils.command.build_ext import build_ext
from distutils.command.build_clib import build_clib
import distutils.sysconfig
import platform
import os
import sys
import shutil
import subprocess
import tempfile
import sysconfig
# from Cython.Build import cythonize

import PyMimircache.const
PyMimircache.const.INSTALL_PHASE = True
from PyMimircache.version import __version__


_DEBUG = False
_DEBUG_LEVEL = 0

# --------------------- Initialization ------------------------------

extensions = []
extra_compile_args = []
extra_link_args = ["-lm"]
numpy_headers = []


if _DEBUG:
    extra_compile_args += ["-g", "-O0", "-D_DEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

# print(sysconfig.get_config_var("CFLAGS").split())
# --------------------- Get OpenMP compiler flag --------------------

# If this C test program compiles the compiler supports OpenMP
# http://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
omp_test = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main()
    {
        #pragma omp parallel
        printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
        return 0;
    }
    """


def get_openmp_flag():
    openmp_flag = ""
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r"omp_test.c"

    with open(filename, "w") as file:
        file.write(omp_test)
        file.flush()

    try:
        cc = os.environ["CC"]
    except KeyError:
        cc = "cc"

    # Compile omp_test.c program using different OpenMP compiler flags.
    # If the code below fails continue without OpenMP support.
    with open(os.devnull, "w") as fnull:
        exit_code = 1
        try:
            exit_code = subprocess.call([cc, "-fopenmp", filename], stdout=fnull, stderr=fnull)
        except:
            pass
        if exit_code == 0:
            openmp_flag = "-fopenmp"
        else:
            try:
                exit_code = subprocess.call([cc, "-openmp", filename], stdout=fnull, stderr=fnull)
            except:
                pass
            if exit_code == 0:
                openmp_flag = "-openmp"

    # clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    return openmp_flag


def get_glib_flag():
    try:
        glib_cflag = subprocess.check_output("pkg-config --cflags glib-2.0", shell=True).decode().strip()
        if not glib_cflag:
            raise RuntimeError("cannot find glib cflag")
        return glib_cflag.split()
    except Exception as e:
        print(e)


def get_glib_library():
    try:
        glib_lib = subprocess.check_output("pkg-config --libs glib-2.0 --libs gthread-2.0", shell=True).decode().strip()
        if not glib_lib:
            raise RuntimeError("cannot find glib lib")
        return glib_lib.split()
    except Exception as e:
        print(e)


def set_platform_related_config():
    if sys.platform == "darwin":
        from distutils import sysconfig
        vars = sysconfig.get_config_vars()
        vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")
        extra_link_args.append("-dynamiclib")
        # extra_compile_args.append("-dynamiclib")
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang"


def get_numpy_header():
    try:
        import numpy
        numpy_headers.append(numpy.get_include())
    except Exception as e:
        print(e)




extra_compile_args.extend(get_glib_flag())
extra_link_args.extend(get_glib_library())
# extra_compile_args.append(get_openmp_flag())
# extra_link_args.append(get_openmp_flag())

numpy_headers.append(get_numpy_header())
set_platform_related_config()


if _DEBUG:
    print("all compile flags: {}".format(extra_compile_args))
    print("all link flasgs: {}".format(extra_link_args))
    print("{}".format(extensions))


BASE_PATH = "PyMimircache/CMimircache/CMimircache/CMimircache/"
COMMON_HEADERS = [BASE_PATH + "/headers", BASE_PATH + "/dataStructure/include"] +  numpy_headers


extensions.append(Extension(
    "PyMimircache.CMimircache.CacheReader",
    glob(BASE_PATH + "/cacheReader/*.c") + glob(BASE_PATH + "/utils/*.c") +
    ["PyMimircache/CMimircache/pyBindings/pyReader.c"],
    include_dirs=[BASE_PATH + "/cacheReader/include", BASE_PATH + "/utils/include/",] + COMMON_HEADERS,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))

extensions.append(Extension(
    "PyMimircache.CMimircache.LRUProfiler",
    [BASE_PATH + "/profiler/LRUProfiler.c", BASE_PATH + "/dataStructure/splay.c",
    BASE_PATH + "/dataStructure/murmur3.c"] +
    glob(BASE_PATH + "/cacheReader/*.c") + glob(BASE_PATH + "/utils/*.c") +
    ["PyMimircache/CMimircache/pyBindings/pyLRUProfiler.c"],
    include_dirs=[BASE_PATH + "/cacheReader/include",
        BASE_PATH + "/profiler/include", BASE_PATH + "/utils/include/",
        "PyMimircache/CMimircache/pyBindings/include"] + COMMON_HEADERS,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))

extensions.append(Extension(
    "PyMimircache.CMimircache.GeneralProfiler",
    glob(BASE_PATH + "/profiler/*.c") + glob(BASE_PATH + "/cache/*.c") +
    glob(BASE_PATH + "/cacheReader/*.c") + glob(BASE_PATH + "/utils/*.c") +
    glob(BASE_PATH + "dataStructure/*.c") +
    ["PyMimircache/CMimircache/pyBindings/pyGeneralProfiler.c",
        "PyMimircache/CMimircache/pyBindings/python_wrapper.c"],
    include_dirs=[BASE_PATH + "/cache/include", BASE_PATH + "/cacheReader/include",
        BASE_PATH + "/profiler/include", BASE_PATH + "/utils/include/",
        "PyMimircache/CMimircache/pyBindings/include"] + COMMON_HEADERS,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))

extensions.append(Extension(
    "PyMimircache.CMimircache.Heatmap",
    glob(BASE_PATH + "/profiler/*.c") + glob(BASE_PATH + "/cache/*.c") +
    glob(BASE_PATH + "/cacheReader/*.c") + glob(BASE_PATH + "/utils/*.c") +
    glob(BASE_PATH + "dataStructure/*.c") +
    ["PyMimircache/CMimircache/pyBindings/python_wrapper.c",
        "PyMimircache/CMimircache/pyBindings/pyHeatmap.c"],
    include_dirs=[BASE_PATH + "/headers"] +
    [BASE_PATH + "/cache/include", BASE_PATH + "/cacheReader/include",
        BASE_PATH + "/profiler/include", BASE_PATH + "/utils/include/"] +
    ["PyMimircache/CMimircache/pyBindings/include"] + COMMON_HEADERS,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))


# extensions.append(Extension(
#     "PyMimircache.c_eviction_stat",
#     glob("PyMimircache/CMimircache/profiler/*.c") +
#     glob("PyMimircache/CMimircache/cache/*.c") +
#     glob("PyMimircache/CMimircache/cacheReader/*.c") +
#     glob("PyMimircache/CMimircache/utils/*.c") +
#     glob("PyMimircache/CMimircache/wrapper/*.c") +
#     ["PyMimircache/CMimircache/pyBindings/python_wrapper.c"] +
#     ["PyMimircache/CMimircache/pyBindings/pyEviction_stat.c"],
#     include_dirs=["PyMimircache/CMimircache/headers"] +
#     ["PyMimircache/CMimircache/cache/include"] +
#     ["PyMimircache/CMimircache/cacheReader/include"] +
#     ["PyMimircache/CMimircache/profiler/include"] +
#     ["PyMimircache/CMimircache/utils/include/"] +
#     ["PyMimircache/CMimircache/pyBindings/include"] +
#     ["PyMimircache/CMimircache/headers/cache"] + numpy_headers,
#     extra_compile_args=extra_compile_args,
#     extra_link_args=extra_link_args,
#     language="c"))





# extensions.append(cythonize("PyMimircache/cache/RR.pyx"))



long_description = ""
try:
    with open("README.md") as f:
        long_description = f.read()
except Exception as e:
    print("failed to read README, {}".format(e))


setup(
    name="PyMimircache",
    version=__version__,
    packages = ["PyMimircache", "PyMimircache.cache", "PyMimircache.cacheReader",
    "PyMimircache.profiler", "PyMimircache.profiler.utils", "PyMimircache.utils", "PyMimircache.top"],
    # modules =
    # package_data={"plain": ["PyMimircache/data/trace.txt"],
    #               "csv": ["PyMimircache/data/trace.csv"],
    #               "vscsi": ["PyMimircache/data/trace.vscsi"],
    #               "conf": ["PyMimircache/conf"]},
    # include_package_data=True,
    author="Juncheng Yang",
    author_email="peter.waynechina@gmail.com",
    description="PyMimircache is a Python3 platform for analyzing cache traces, "
                "developed by Juncheng Yang in Ymir group @ Emory University",
    license="GPLv3",
    keywords="PyMimircache cache LRU simulator Emory Ymir",
    url="http://mimircache.info",

    ext_modules=extensions,
    install_requires=["heapdict", "mmh3", "matplotlib", "numpy"]
)


# CC="ccache gcc" CXX="ccache" python3 setup.py build_ext -i


# python3 setup.py sdist upload -r pypitest

# # Compile into .o files
# objects = c.compile(["a.c", "b.c"])

# # Create static or shared library
# c.create_static_lib(objects, "foo", output_dir=workdir)
# c.link_shared_lib(objects, "foo", output_dir=workdir)
