//
//  python_wrapper.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>
#include "LRUProfiler.h"

#define NPY_NO_DEPRECATED_API 11
#include <numpy/arrayobject.h>


/* TODO: 
not urgent, not necessary: change this profiler module into a pyhton object, 
    this is not necessary for now because we are not going to expose this level API 
    to user, instead we wrap it with our python API, so these C functions are only 
    called inside mimircache  
*/



static PyObject* LRUProfiler_get_reuse_dist_seq(PyObject* self,
                                                PyObject* args,
                                                PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 begin=-1, end=-1;
    static char *kwlist[] = {"reader", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &po)) {
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (begin==-1)
        begin = 0;
    
    // get reuse dist
    gint64* reuse_dist = get_reuse_dist_seq(reader, begin, end);
    
    // create numpy array
    if (begin < 0)
        begin = 0;
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    if (end < 0)
        end = reader->base->total_num;
    
    npy_intp dims[1] = { end-begin };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
    guint64 i;
    for (i=0; i<(guint64)(end-begin); i++)
        *((long long*)PyArray_GETPTR1((PyArrayObject*)ret_array, i)) = (long long)reuse_dist[i];
    
    
    return ret_array;
}


static PyObject* LRUProfiler_get_future_reuse_dist(PyObject* self,
                                                   PyObject* args,
                                                   PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 begin=-1, end=-1;
    static char *kwlist[] = {"reader", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist, &po)) {
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // get reuse dist
    gint64* reuse_dist = get_future_reuse_dist(reader, begin, end);
    
    // create numpy array
    if (begin < 0)
        begin = 0;
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    if (end < 0)
        end = reader->base->total_num;
    
    npy_intp dims[1] = { end-begin };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
    //    memcpy(PyArray_DATA((PyArrayObject*)ret_array), reuse_dist, sizeof(long long)*(end-begin));
    guint64 i;
    for (i=0; i<(guint64)(end-begin); i++)
        *((long long*)PyArray_GETPTR1((PyArrayObject*)ret_array, i)) = (long long)reuse_dist[i];
    
    return ret_array;
}



static PyObject* LRUProfiler_load_reuse_dist(PyObject* self,
                                             PyObject* args,
                                             PyObject* keywds)
{
    PyObject* po;
    char *cp_file_loc, *cp_rd_type;
    int rd_type;
    
    reader_t* reader;

    static char *kwlist[] = {"reader", "file_loc", "rd_type", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oss", kwlist,
                                     &po, &cp_file_loc, &cp_rd_type)) {
        return NULL;
    }

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (strcmp(cp_rd_type, "rd") == 0)
        rd_type = NORMAL_REUSE_DISTANCE;
    else if (strcmp(cp_rd_type, "frd") == 0)
        rd_type = FUTURE_REUSE_DISTANCE;
    else{
        ERROR("error with rd_type\n");
        abort();
    }
    

    // load reuse dist
    load_reuse_dist(reader, cp_file_loc, rd_type);
    
    Py_RETURN_NONE;
}


static PyObject* LRUProfiler_save_reuse_dist(PyObject* self,
                                             PyObject* args,
                                             PyObject* keywds)
{
    PyObject* po;
    char *cp_file_loc, *cp_rd_type;
    int rd_type = NORMAL_REUSE_DISTANCE; 
    
    reader_t* reader;
    
    static char *kwlist[] = {"reader", "file_loc", "rd_type", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oss", kwlist,
                                     &po, &cp_file_loc, &cp_rd_type)) {
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        ERROR("error retrieving reader pointer from PyCapsule\n");
        abort();
    }
    
    
    if (strcmp(cp_rd_type, "rd") == 0)
        rd_type = NORMAL_REUSE_DISTANCE;
    else if (strcmp(cp_rd_type, "frd") == 0)
        rd_type = FUTURE_REUSE_DISTANCE;
    else{
        ERROR("error with rd_type\n");
        abort();
    }
    cal_save_reuse_dist(reader, cp_file_loc, rd_type);
    
    
    Py_RETURN_NONE;
}




static PyObject* LRUProfiler_get_hit_count_seq(PyObject* self, PyObject* args, PyObject* keywds)
{   
    PyObject* po;
    reader_t* reader;
    gint64 cache_size = -1;
    gint64 begin=-1, end=-1;
    static char *kwlist[] = {"reader", "cache_size", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|lll", kwlist,
                                &po, &cache_size, &begin, &end)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (begin == -1)
        begin = 0;
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    if (end == -1)
        end = reader->base->total_num;
    
    // get hit count 
    guint64* hit_count = get_hit_count_seq(reader, cache_size, begin, end);

    // create numpy array 
    if (cache_size == -1){
        cache_size = reader->base->total_num;
    }
    if (end-begin < cache_size)
        cache_size = end - begin;

    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
    guint64 i;
    for (i=0; i<(guint64)cache_size+3; i++){
        *((long long*)PyArray_GETPTR1((PyArrayObject*)ret_array, i)) = (long long)hit_count[i];
    }


    g_free(hit_count);

    return ret_array;
}



static PyObject* LRUProfiler_get_hit_rate_seq(PyObject* self, PyObject* args, PyObject* keywds)
{   
    PyObject* po;
    reader_t* reader;
    gint64 cache_size=-1;
    gint64 begin=-1, end=-1;
    static char *kwlist[] = {"reader", "cache_size", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|lll", kwlist,
                                &po, &cache_size, &begin, &end)) {
        return NULL;
    }

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    if (begin == -1)
        begin = 0;
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    if (end == -1)
        end = reader->base->total_num;
    
    // get hit rate
    double* hit_rate = get_hit_rate_seq(reader, cache_size, begin, end);

    // create numpy array
    if (cache_size == -1)
        cache_size = reader->base->total_num;
    if (end-begin < cache_size)
        cache_size = end - begin;
    

    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)ret_array), hit_rate, sizeof(double)*(cache_size+3));

    if (!(begin==0 && (end==-1 || end==reader->base->total_num))){
        g_free(hit_rate);
    }

    return ret_array;
}

static PyObject* LRUProfiler_get_miss_rate_seq(PyObject* self,
                                               PyObject* args,
                                               PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 cache_size=-1;
    gint64 begin=-1, end=-1;
    static char *kwlist[] = {"reader", "cache_size", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|lll", kwlist,
                                &po, &cache_size, &begin, &end)) {
        return NULL;
    }

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (begin == -1)
        begin = 0;
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    if (end == -1)
        end = reader->base->total_num;

    
    // get miss rate
    double* miss_rate = get_miss_rate_seq(reader, cache_size, begin, end);

    // create numpy array 
    if (cache_size == -1)
        cache_size = reader->base->total_num;
    if (end-begin < cache_size)
        cache_size = end - begin;
    

    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)ret_array), miss_rate, sizeof(double)*(cache_size+3));
    g_free(miss_rate);

    return ret_array;
}




static PyObject* LRUProfiler_get_hit_rate_seq_shards(PyObject* self,
                                                     PyObject* args,
                                                     PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 cache_size=-1, correction=0;
    double sample_ratio;
    static char *kwlist[] = {"reader", "sample_ratio", "cache_size", "correction", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Od|ll", kwlist,
                                     &po, &sample_ratio, &cache_size, &correction)) {
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    // get hit rate
    double* hit_rate = get_hit_rate_seq_shards(reader, cache_size,
                                               sample_ratio, correction);
    
    // create numpy array
    if (cache_size == -1){
        cache_size = (gint64)(reader->base->total_num / sample_ratio);
    }
    
    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)ret_array), hit_rate, sizeof(double)*(cache_size+3));
    
    
    return ret_array;
}


static PyObject* LRUProfiler_get_hit_rate_withsize_seq(PyObject* self,
                                                     PyObject* args,
                                                     PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 cache_size=-1;
    int block_size=0;
    static char *kwlist[] = {"reader", "cache_size", "block_unit_size", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|li", kwlist,
                                     &po, &cache_size, &block_size)) {
        return NULL;
    }
    
    if (block_size == 0){
        ERROR("block size 0\n");
        exit(1);
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    // get hit rate
    double* hit_rate = get_hitrate_withsize_seq(reader, cache_size, block_size);

    
    // create numpy array
    if (cache_size == -1){
        cache_size = (gint64)(reader->base->total_num);
    }
    
    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*)ret_array), hit_rate, sizeof(double)*(cache_size+3));
    
    
    return ret_array;
}


static PyObject* LRUProfiler_get_hit_count_withsize_seq(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    gint64 cache_size = -1;
    int block_size = 0;
    static char *kwlist[] = {"reader", "cache_size", "block_unit_size", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|li", kwlist,
                                     &po, &cache_size, &block_size)) {
        return NULL;
    }
    if (block_size == 0){
        ERROR("block size 0\n");
        exit(1);
    }

    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    // get hit count
    guint64* hit_count = get_hitcount_withsize_seq(reader, cache_size, block_size);
    
    // create numpy array
    if (cache_size == -1)
        cache_size = reader->base->total_num;
    
    
    npy_intp dims[1] = { cache_size+3 };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
    guint64 i;
    for (i=0; i<(guint64)cache_size+3; i++){
        *((long long*)PyArray_GETPTR1((PyArrayObject*)ret_array, i)) = (long long)hit_count[i];
    }
    
    g_free(hit_count);
    
    return ret_array;
}




static PyMethodDef c_LRUProfiler_funcs[] = {
    {"get_hit_count_seq", (PyCFunction)LRUProfiler_get_hit_count_seq,
        METH_VARARGS | METH_KEYWORDS, "get hit count array in the form of numpy array, \
        the last one is cold miss, the second to last is out of cache_size"},
    {"get_hit_rate_seq", (PyCFunction)LRUProfiler_get_hit_rate_seq,
        METH_VARARGS | METH_KEYWORDS, "get hit rate array in the form of numpy array, \
        the last one is cold miss, the second to last is out of cache_size"},
    {"get_miss_rate_seq", (PyCFunction)LRUProfiler_get_miss_rate_seq,
        METH_VARARGS | METH_KEYWORDS, "get miss rate array in the form of numpy array, \
        the last one is cold miss, the second to last is out of cache_size"},
    {"get_future_reuse_dist", (PyCFunction)LRUProfiler_get_future_reuse_dist,
        METH_VARARGS | METH_KEYWORDS, "get reuse distance array of the reversed trace file in the form of numpy array"},
    {"get_reuse_dist_seq", (PyCFunction)LRUProfiler_get_reuse_dist_seq,
        METH_VARARGS | METH_KEYWORDS, "get reuse distance array in the form of numpy array"},
    {"save_reuse_dist", (PyCFunction)LRUProfiler_save_reuse_dist,
        METH_VARARGS | METH_KEYWORDS, "save reuse distance array to specified file"},
    {"load_reuse_dist", (PyCFunction)LRUProfiler_load_reuse_dist,
        METH_VARARGS | METH_KEYWORDS, "load reuse distance array from specified file"},

    
    
    
    {"get_hit_rate_seq_shards", (PyCFunction)LRUProfiler_get_hit_rate_seq_shards,
        METH_VARARGS | METH_KEYWORDS, "shards version"},
    {"get_hit_rate_with_size", (PyCFunction)LRUProfiler_get_hit_rate_withsize_seq,
        METH_VARARGS | METH_KEYWORDS, "LRU profiler consider request size"},
    {"get_hit_count_with_size", (PyCFunction)LRUProfiler_get_hit_count_withsize_seq,
        METH_VARARGS | METH_KEYWORDS, "LRU profiler consider request size"},
    
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_LRUProfiler_definition = { 
    PyModuleDef_HEAD_INIT,
    "c_LRUProfiler",
    "A Python module that doing profiling with regards to LRU cache",
    -1, 
    c_LRUProfiler_funcs
};



PyMODINIT_FUNC PyInit_c_LRUProfiler(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&c_LRUProfiler_definition);
}

