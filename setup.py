# from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
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

# --------------------- Initialization ------------------------------

extensions = []
extra_compile_args = []
extra_link_args = []

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

def get_compiler_openmp_flag():
    openmp_flag = ''
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'omp_test.c'

    with open(filename, 'w') as file:
        file.write(omp_test)
        file.flush()

    try:
        cc = os.environ['CC']
    except KeyError:
        cc = 'cc'
    # os.environ["CC"] = "gcc" 
    # os.environ["CXX"] = "gcc"    

    # Compile omp_test.c program using different OpenMP compiler flags.
    # If the code below fails continue without OpenMP support.
    with open(os.devnull, 'w') as fnull:
        exit_code = 1
        try:
            exit_code = subprocess.call([cc, '-fopenmp', filename], stdout=fnull, stderr=fnull)
        except:
            pass
        if exit_code == 0:
            openmp_flag = '-fopenmp'
        else:
            try:
                exit_code = subprocess.call([cc, '-openmp', filename], stdout=fnull, stderr=fnull)
            except:
                pass
            if exit_code == 0:
                openmp_flag = '-openmp'

    #clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)
    return openmp_flag


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

def setPlatformRelatedConfig():
    if sys.platform == 'darwin':
        from distutils import sysconfig
        vars = sysconfig.get_config_vars()
        vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
        extra_link_args.append('-dynamiclib')
        # extra_compile_args.append('-dynamiclib')
        os.environ["CC"] = "gcc" 
        os.environ["CXX"] = "gcc"

getGlibFlag()
getGlibLibrary()
setPlatformRelatedConfig()
openmp_flag = get_compiler_openmp_flag()
print('get_compiler_openmp_flag(): ' + openmp_flag)

extra_compile_args.append(openmp_flag)
extra_link_args.append(openmp_flag)

# ------------------ OpenMP hack for build_clib --------------------

# if openmp_flag != '':
#     try:
#         cflags = distutils.sysconfig.get_config_var('CFLAGS')
#         distutils.sysconfig._config_vars['CFLAGS'] = cflags + " " + openmp_flag
#
#         # ldflags = distutils.sysconfig.get_config_var('LDFLAGS')
#         # distutils.sysconfig._config_vars['LDFLAGS'] = ldflags + " " + openmp_flag
#
#         # pycflags = distutils.sysconfig.get_config_var('PY_CFLAGS')
#         # distutils.sysconfig._config_vars['PY_CFLAGS'] = pycflags + " " + openmp_flag
#
#         # config_CC = distutils.sysconfig.get_config_var('CC')
#         # distutils.sysconfig._config_vars['CC'] = config_CC + " " + openmp_flag
#
#     except Exception as e:
#         print(e)
#         cflags += openmp_flag


print('all compile flags: ' + str(extra_compile_args))
print('all link flasgs: ' + str(extra_link_args))


# --------------------- parda module ---------------------------
extensions.append(Extension(
    "mimirCache.Profiler.libparda",
    glob("mimirCache/Profiler/parda/src/*.c"),
    include_dirs=["mimirCache/Profiler/parda/src/h/"],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    language = "c"
    ))

# --------------------- vscsi module ---------------------------
extensions.append(Extension(
    "mimirCache.CacheReader.libvscsi",
    glob("mimirCache/CacheReader/vscsi/src/*.c"),
    include_dirs=["mimirCache/CacheReader/vscsi/src/"],
    language = "c"
    ))



# --------------------- parda module in c_lib---------------------------
# parda_include = []
# for i in extra_compile_args:
#     if i.startswith('-I'):
#         parda_include.append(i[2:])
# parda_include.append("cachecow/Profiler/parda/src/h/")
# print(parda_include)

# libparda = ("cachecow.Profiler.libparda", dict(
#     sources = glob("cachecow/Profiler/parda/src/*.c"),
#     include_dirs = parda_include,
#     extra_compile_args = extra_compile_args,
#     # extra_link_args = extra_link_args,
#     language = "c"
#     ))

# --------------------- vscsi module in c_lib---------------------------
# libvscsi = ("cachecow.CacheReader.libvscsi", dict(
#     sources = glob("cachecow/CacheReader/vscsi/src/*.c"),
#     include_dirs = ["cachecow/CacheReader/vscsi/src/"],
#     language = "c"
#     ))




print("find packages: " + str(find_packages(exclude=(['mimirCache.Bin', 'mimirCache.Test', 'mimirCache.Data']))))


setup(
    name="mimirCache",
    version="0.0.1",
    # package_dir = {'':'src'},
    # packages = ['Cache', 'CacheReader', 'Profiler', 'Utils'],
    packages=find_packages(exclude=(['mimirCache.Bin', 'mimirCache.Test', 'mimirCache.Data'])),
    # modules = 
    package_data = {'':['Data/*.trace']}, 

    author = "Juncheng Yang", 
    author_email = "peter.waynechina@gmail.com",
    description="mimirCache platform for analyzing cache traces, developed by Ymir group @ Emory University",
    license = "GPLv3",
    keywords="mimirCache cache Ymir",
    # install_requires = ['scipy', 'numpy', 'matplotlib'], 


    # libraries = [libparda, libvscsi],
    # cmdclass = {'build_clib' : build_clib},
    ext_modules = extensions,
    # cmdclass = {'build_ext': build_ext_subclass, 'build_clib' : build_clib_subclass},
    classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    ],
)





# could also try this: 
# from distutils.ccompiler import new_compiler

# # Create compiler with default options
# c = new_compiler()
# workdir = "."

# # Optionally add include directories etc.
# # c.add_include_dir("some/include")

# # Compile into .o files
# objects = c.compile(["a.c", "b.c"])

# # Create static or shared library
# c.create_static_lib(objects, "foo", output_dir=workdir)
# c.link_shared_lib(objects, "foo", output_dir=workdir)