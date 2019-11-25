//
//  python_wrapper.h
//  pyMimircache
//
//  Created by Juncheng on 5/24/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//

#ifndef python_wrapper_h
#define python_wrapper_h




// python3-config --include
// np.get_include()

#define NPY_NO_DEPRECATED_API 11


#if defined(__linux__) || defined(__APPLE__)
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
//#elif __APPLE__
//#include "/usr/local/Cellar/python/3.7.5/Frameworks/Python.framework/Versions/3.7/include/python3.7m/Python.h"
//#include "/Users/jason/Library/Python/3.7/lib/python/site-packages/numpy/core/include/numpy/arrayobject.h"
//#include "/Users/jason/Library/Python/3.7/lib/python/site-packages/numpy/core/include/numpy/npy_math.h"
#elif _WIN32
#warning "windows unsupported"
#endif




#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <math.h>

#include "../libMimircache/libMimircache/include/mimircache.h"



cache_t* build_cache(reader_t* reader,
                          long cache_size,
                          char* algorithm,
                          PyObject* cache_params,
                          long begin);




#endif /* python_wrapper_h */



