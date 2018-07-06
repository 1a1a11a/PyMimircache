//
//  python_wrapper.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>

#define NPY_NO_DEPRECATED_API 11
#include <numpy/arrayobject.h>

#include "generalProfiler.h"
#include "partition.h"
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include <math.h>
#include "python_wrapper.h"
#include "logging.h"


/* TODO:
 not urgent, not necessary: change this profiler module into a pyhton object,
 this is not necessary for now because we are not going to expose this level API
 to user, instead we wrap it with our python API, so these C functions are only
 called inside mimircache
 */


static PyObject* generalProfiler_get_hit_rate(PyObject* self,
                                              PyObject* args,
                                              PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    int bin_size = -1;
    char* algorithm;
    struct_cache* cache;
    PyObject* cache_params;

    long begin=0, end=-1;
    static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
        "cache_params", "num_of_threads", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oill", kwlist, &po,
                                     &algorithm, &cache_size, &bin_size,
                                     &cache_params, &num_of_threads, &begin, &end)) {
        ERROR("parsing argument failed in generalProfiler_get_hit_rate\n");
        return NULL;
    }

    if(begin == -1)
        begin = 0;

    DEBUG("bin size: %d, threads: %d\n", bin_size, num_of_threads);

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, begin);

    // get hit rate
    DEBUG("before profiling\n");
    return_res_t** results = profiler(reader, cache, num_of_threads, bin_size,
                                    (gint64)begin, (gint64)end);
    DEBUG("after profiling\n");

    // create numpy array
    guint num_of_bins = ceil((double) cache_size/bin_size)+1;
    npy_intp dims[1] = { num_of_bins };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    guint i;
    *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, 0) = 0;
    for(i=0; i<num_of_bins; i++){
        *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, i) = results[i]->hit_rate;
        g_free(results[i]);
    }

//    PyObject *d = PyDict_New();
//    for (i=0; i<num_of_bins; i++){
//        PyDict_SetItem(d, Py_BuildValue("l", (i+1)*bin_size), Py_BuildValue("f", results[i]->hit_rate));
//    }

    g_free(results);
    cache->core->destroy(cache);
    return ret_array;
}






static PyObject* generalProfiler_get_hit_count(PyObject* self,
                                               PyObject* args,
                                               PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    int bin_size = -1;
    char* algorithm;
    struct_cache* cache;
    PyObject* cache_params=NULL;

    long begin=0, end=-1;
    static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
        "cache_params", "num_of_threads", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oill", kwlist, &po,
                                     &algorithm, &cache_size, &bin_size,
                                     &cache_params, &num_of_threads, &begin, &end)) {
        printf("parsing argument failed in generalProfiler_get_hit_rate\n");
        return NULL;
    }

    if(begin == -1)
        begin = 0;

    DEBUG("bin size: %d, threads: %d\n", bin_size, num_of_threads);

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, begin);


    // get hit rate
    DEBUG("before profiling\n");
    return_res_t** results = profiler(reader, cache, num_of_threads, bin_size,
                                    (gint64)begin, (gint64)end);
    DEBUG("after profiling\n");

    // create numpy array
    guint num_of_bins = ceil((double) cache_size/bin_size)+1;
    npy_intp dims[1] = { num_of_bins };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
    guint64 i;
    *(long long *)PyArray_GETPTR1((PyArrayObject*)ret_array, 0) = 0;
    for(i=0; i<num_of_bins; i++){
        *(long long *)PyArray_GETPTR1((PyArrayObject*)ret_array, i) = results[i]->hit_count;
        g_free(results[i]);
    }


    g_free(results);
    cache->core->destroy(cache);
    return ret_array;
}


static PyObject* generalProfiler_get_miss_rate(PyObject* self,
                                               PyObject* args,
                                               PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    int bin_size = -1;
    char* algorithm;
    struct_cache* cache;
    PyObject* cache_params=NULL;

    long begin=0, end=-1;
    static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
        "cache_params", "num_of_threads", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oill", kwlist, &po,
                                     &algorithm, &cache_size, &bin_size,
                                     &cache_params, &num_of_threads, &begin, &end)) {
        printf("parsing argument failed in generalProfiler_get_hit_rate\n");
        return NULL;
    }

    if(begin == -1)
        begin = 0;

    DEBUG("bin size: %d, threads: %d\n", bin_size, num_of_threads);

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, begin);


    // get hit rate
    DEBUG("before profiling\n");
    return_res_t** results = profiler(reader, cache, num_of_threads, bin_size,
                                    (gint64)begin, (gint64)end);
    DEBUG("after profiling\n");

    // create numpy array
    guint num_of_bins = ceil((double) cache_size/bin_size)+1;
    npy_intp dims[1] = { num_of_bins };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    guint i;
    *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, 0) = 0;
    for(i=0; i<num_of_bins; i++){
        *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, i) = results[i]->miss_rate;
        g_free(results[i]);
    }


    g_free(results);
    cache->core->destroy(cache);
    return ret_array;
}


static PyObject* generalProfiler_get_evict_err_rate(PyObject* self,
                                                    PyObject* args,
                                                    PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    char* mode;
    long cache_size;
    guint64 time_interval;
    char* algorithm = "LRU";
    struct_cache* cache;
    PyObject* cache_params;

    static char *kwlist[] = {"reader", "mode", "time_interval", "cache_size",
        "algorithm", "cache_params", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oslls|O", kwlist, &po,
                                     &mode, &time_interval, &cache_size,
                                     &algorithm, &cache_params)) {
        fprintf(stderr, "parsing argument failed in generalProfiler_get_hit_rate\n");
        return NULL;
    }

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, 0);


    // get hit rate
    gdouble* result = LRU_evict_err_statistics(reader, cache, time_interval);

    // create numpy array
    guint num_of_bins = reader->sdata->break_points->array->len - 1;
    npy_intp dims[1] = { num_of_bins };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    guint i;
    *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, 0) = 0;
    for(i=0; i<num_of_bins; i++){
        *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, i) = (double)result[i];
    }

    cache->core->destroy(cache);
    return ret_array;
}






static PyObject* generalProfiler_get_partition(PyObject* self,
                                               PyObject* args,
                                               PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    long cache_size;
    char* algorithm;
    int n_partitions;
    struct_cache* cache;
    PyObject* cache_params=NULL;

    static char *kwlist[] = {"reader", "algorithm", "cache_size",
        "n_partitions", "cache_params", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|O", kwlist, &po,
                                     &algorithm, &cache_size, &n_partitions,
                                     &cache_params)) {
        fprintf(stderr, "parsing argument failed in generalProfiler_get_hit_rate\n");
        return NULL;
    }


    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, 0);


    // get partition
    DEBUG("partitions %i, %ld before getting partition\n", n_partitions, cache_size);
    partition_t* partitions = get_partition(reader, cache, n_partitions);
    DEBUG("after getting partition\n");

    // create numpy array
    npy_intp dims[2] = { n_partitions, partitions->partition_history[0]->len };
    PyObject* ret_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    gint i, j;

    double *array;
    gint length = (gint) partitions->partition_history[0]->len;
    for (i=0; i<n_partitions; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<length; j++)
            array[j] = g_array_index(partitions->partition_history[i], double, j);
    }



    free_partition_t(partitions);
    cache->core->destroy(cache);
    return ret_array;
}



static PyObject* generalProfiler_get_partition_hit_rate(PyObject* self,
                                                        PyObject* args,
                                                        PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    int bin_size = -1;
    char* algorithm;
    struct_cache* cache;
    PyObject* cache_params;

    static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
        "cache_params", "num_of_threads", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oi", kwlist, &po,
                                     &algorithm, &cache_size, &bin_size,
                                     &cache_params, &num_of_threads)) {
        fprintf(stderr, "parsing argument failed in generalProfiler_get_partition_hit_rate\n");
        return NULL;
    }


    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // build cache
    cache = build_cache(reader, cache_size, algorithm, cache_params, 0);

    // get hit rate
    return_res_t** results = profiler_partition(reader, cache, num_of_threads, bin_size);

    // create numpy array
    guint num_of_bins = ceil((double) cache_size/bin_size)+1;
    npy_intp dims[1] = { num_of_bins };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    guint i;
    *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, 0) = 0;
    for(i=0; i<num_of_bins; i++){
        *(double*)PyArray_GETPTR1((PyArrayObject*)ret_array, i) = results[i]->hit_rate;
        g_free(results[i]);
    }

    g_free(results);
    cache->core->destroy(cache);
    return ret_array;
}











static PyMethodDef GeneralProfiler_funcs[] = {
    {"get_hit_ratio", (PyCFunction)generalProfiler_get_hit_rate,
        METH_VARARGS | METH_KEYWORDS, "get hit rate numpy array"},
    {"get_hit_count", (PyCFunction)generalProfiler_get_hit_count,
        METH_VARARGS | METH_KEYWORDS, "get hit count numpy array"},
    {"get_miss_ratio", (PyCFunction)generalProfiler_get_miss_rate,
        METH_VARARGS | METH_KEYWORDS, "get miss rate numpy array"},
    {"get_partition", (PyCFunction)generalProfiler_get_partition,
        METH_VARARGS | METH_KEYWORDS, "get partition results with given alg"},
    {"get_partition_hit_ratio", (PyCFunction)generalProfiler_get_partition_hit_rate,
        METH_VARARGS | METH_KEYWORDS, "get partition hit rate numpy array"},


    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef GeneralProfiler_definition = {
    PyModuleDef_HEAD_INIT,
    "GeneralProfiler",
    "A Python module that doing profiling with regards to all kinds of caches",
    -1,
    GeneralProfiler_funcs
};



PyMODINIT_FUNC PyInit_GeneralProfiler(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&GeneralProfiler_definition);
}
