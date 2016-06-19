from glob import glob
import subprocess

from distutils.core import setup, Extension

ext = [Extension('helloworld', ['cacheReader/hello.c']),
       ]

extra_compile_args = ['-Wall', '-O0', '-g']
extra_link_args = []
numpy_headers = []


def getGlibFlag():
    try:
        glib_cflag = subprocess.check_output("pkg-config --cflags glib-2.0", shell=True).decode().strip()
        extra_compile_args.extend(glib_cflag.split())
    except Exception as e:
        print(e)


def getGlibLibrary():
    try:
        glib_lib = subprocess.check_output("pkg-config --libs glib-2.0 --libs gthread-2.0", shell=True).decode().strip()
        extra_link_args.extend(glib_lib.split())
    except Exception as e:
        print(e)


def getNumpyHeader():
    try:
        import numpy
        numpy_headers.append(numpy.get_include())
    except Exception as e:
        print(e)


getGlibFlag()
getGlibLibrary()
getNumpyHeader()

ext.append(Extension(
    'c_cacheReader',
    glob("cacheReader/*.c"),
    include_dirs=["headers"] + numpy_headers,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))

ext.append(Extension(
    'c_LRUProfiler',
    glob("profiler/LRUProfiler/*.c") + ['utils/glib_related.c'] + ['cacheReader/reader.c'],
    include_dirs=["headers"] + numpy_headers,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c"))

setup(name='helloworld', version='1.0',
      ext_modules=ext,
      )
