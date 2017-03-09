//
//  python_wrapper_eviction_stat.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>
#include "eviction_stat.h"
#include "cache.h"
#include "FIFO.h" 
#include "Optimal.h"
#include "const.h"
#include "python_wrapper.h"


#define NPY_NO_DEPRECATED_API 11
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>




static PyObject* eviction_stat_get_stat(PyObject* self, PyObject* args, PyObject* keywds)
{   
    PyObject *po, *cache_params=NULL;
    reader_t* reader;
    long cache_size;
    char *algorithm, *stat_type_s;
    struct_cache* cache;
    evict_stat_type stat_type;
    static char *kwlist[] = {"reader", "algorithm", "cache_size", "stat_type", "cache_params", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osls|O", kwlist, &po, &algorithm, &cache_size, &stat_type_s, &cache_params)) {
        // currently specifying begin and ending position is not supported 
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, 0);
    
    
    if (strcmp(stat_type_s, "freq") == 0)
        stat_type = evict_freq;
    else if (strcmp(stat_type_s, "accumulative_freq") == 0)
        stat_type = evict_freq_accumulatve;
    else if (strcmp(stat_type_s, "reuse_dist") == 0)
        stat_type = evict_reuse_dist;
    else if (strcmp(stat_type_s, "data_classification") == 0)
        stat_type = evict_data_classification;
    else {
        printf("unsupported stat type\n");
        exit(1);
    }

    
    
    // get eviction stat array
    gint64* evicetion_stat_array = eviction_stat(reader, cache, stat_type);

    // create numpy array
    long long size = reader->base->total_num;

    npy_intp dims[1] = { size };
    PyObject* ret_array;
    
//    if (stat_type != evict_freq_relative){
        ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
        long long* array = (long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0);
        int i;
        for (i=0; i<reader->base->total_num; i++)
            array[i] = evicetion_stat_array[i];
//    }
//    else{
//        ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
//        double* array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0);
//        int i;
//        for (i=0; i<reader->total_num; i++)
//            array[i] = ((gdouble*)evicetion_stat_array)[i];
//    }
    
    if (cache->core->destroy)
        // if it is LRU, then it doesn't have a destroy function
        cache->core->destroy(cache);
    else
        cache_destroy(cache);
    
    g_free(evicetion_stat_array);
    
    
    return ret_array;
}






static PyMethodDef c_eviction_stat_funcs[] = {
    {"get_stat", (PyCFunction)eviction_stat_get_stat,
        METH_VARARGS | METH_KEYWORDS, "get the statistics about the evicted item, -1 means \
        at given timestamp, no item is evicted"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_eviction_stat_definition = {
    PyModuleDef_HEAD_INIT,
    "c_eviction_stat",
    "A C module that doing statistics for the evicted requests",
    -1, 
    c_eviction_stat_funcs
};



PyMODINIT_FUNC PyInit_c_eviction_stat(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&c_eviction_stat_definition);
}

