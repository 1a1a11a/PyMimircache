//
//  python_wrapper.h
//  python_wrapper
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef python_wrapper_h
#define python_wrapper_h




#include <Python.h>

#define NPY_NO_DEPRECATED_API 11

#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "LFU.h" 
#include "LRU_LFU.h" 
#include "LRU_dataAware.h" 

#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include "reader.h"
#include "glib_related.h"
#include "cache.h"
#include "const.h"


struct_cache* build_cache(READER* reader, long cache_size, char* algorithm, PyObject* cache_params, long begin);




#endif /* python_wrapper_h */