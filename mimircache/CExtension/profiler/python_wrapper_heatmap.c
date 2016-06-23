//
//  python_wrapper.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>
#include "heatmap.h"
#include "cache.h"
#include "FIFO.h" 
#include "Optimal.h"
#include "const.h"


#define NPY_NO_DEPRECATED_API 11
#include <numpy/arrayobject.h>


/* TODO: 
not urgent, not necessary: change this profiler module into a python object,
    this is not necessary for now because we are not going to expose this level API 
    to user, instead we wrap it with our python API, so these C functions are only 
    called inside mimircache  
*/


static PyObject* differential_heatmap_py(PyObject* self, PyObject* args, PyObject* keywds);




static PyObject* heatmap_get_last_access_dist_seq(PyObject* self, PyObject* args, PyObject* keywds)
{   
    PyObject* po;
    READER* reader; 
    long begin=-1, end=-1; 
    static char *kwlist[] = {"reader", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist, 
                                &po, &begin, &end)) {
        // currently specifying begin and ending position is not supported 
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // get last access dist list  
    GSList* list = get_last_access_dist_seq(reader, read_one_element);

    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (begin == -1)
        begin = 0;
    if (end == -1)
        end = reader->total_num;

    // create numpy array
    long long size = end - begin;

    npy_intp dims[1] = { size };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_INT); 
    GSList* gsl;
    long long counter = size-1;
    int* array = (int*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0); 
    for (gsl=list; gsl!=NULL; gsl=gsl->next){
        array[counter--] = GPOINTER_TO_INT(gsl->data);
    }

    // memcpy(PyArray_DATA((PyArrayObject*)ret_array), hit_count, sizeof(long long)*(cache_size+3));
    g_slist_free(list); 
    
    return ret_array;
}


static PyObject* heatmap_get_next_access_dist_seq(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    READER* reader;
    long begin=-1, end=-1;
    static char *kwlist[] = {"reader", "begin", "end", NULL};
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist,
                                     &po, &begin, &end)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // get reversed last access dist list
    GSList* list = get_last_access_dist_seq(reader, read_one_element_above);
    
    if (reader->total_num == -1)
        get_num_of_cache_lines(reader);

    if (begin == -1)
        begin = 0;
    if (end == -1)
        end = reader->total_num;

    // create numpy array
    long long size = end - begin;
    npy_intp dims[1] = { size };
    PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_INT);
    GSList* gsl = list;
    long long counter = 0;
    int* array = (int*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0);
    long i;
    for (i=0; i<begin; i++)
        gsl = gsl->next;
    
    
    for (i=begin; i<end; i++,gsl=gsl->next){
        array[counter++] = GPOINTER_TO_INT(gsl->data);
    }
    g_slist_free(list);
    
    return ret_array;
}


static PyObject* heatmap_computation(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    READER* reader;
    int num_of_threads = 4;
    long cache_size;
    char* name;
    char* plot_type_s;
    int plot_type;
    char* mode;
    long time_interval;
    struct_cache* cache;
    
    static char *kwlist[] = {"reader", "cache_size", "cache_type", "mode", "time_interval", "plot_type", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Olssls|$i", kwlist, &po, &cache_size, &name, &mode, &time_interval, &plot_type_s, &num_of_threads)) {
        printf("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    
    printf("plot type: %s, cache size: %ld, mode: %s, time_interval: %ld, num_of_threads: %d\n", plot_type_s, cache_size, mode, time_interval, num_of_threads);
    
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache
    char data_type = reader->type;
    
    if (strcmp(name, "FIFO") == 0){
        cache = fifo_init(cache_size, data_type, NULL);
    }
    
    else if (strcmp(name, "Optimal") == 0){
        struct optimal_init_params init_params = {.reader=reader, .next_access=NULL};
        cache = optimal_init(cache_size, data_type, (void*)&init_params);
    }
    else if (strcmp(name, "LRU") == 0){
        cache = cache_init(cache_size, data_type);
        cache->core->type = e_LRU;
    }
    else {
        printf("does not support given cache replacement algorithm: %s\n", name);
        exit(1);
    }

    if (strcmp(plot_type_s, "hit_rate_start_time_end_time") == 0)
        plot_type = hit_rate_start_time_end_time;
    else if (strcmp(plot_type_s, "hit_rate_start_time_cache_size") == 0)
        plot_type = hit_rate_start_time_cache_size;
    else if (strcmp(plot_type_s, "avg_rd_start_time_end_time") == 0)
        plot_type = avg_rd_start_time_end_time;
    else if (strcmp(plot_type_s, "cold_miss_count_start_time_end_time") == 0)
        plot_type = cold_miss_count_start_time_end_time;
    else if (strcmp(plot_type_s, "rd_distribution") == 0){
        printf("please use function heatmap_rd_distribution\n");
        Py_RETURN_NONE;
    }
    else {
        printf("unsupported plot type\n");
        exit(1);
    }
    
    printf("before computation\n");
    draw_dict* dd = heatmap(reader, cache, *mode, time_interval, plot_type, num_of_threads);
    printf("after computation\n");
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };

    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);

    
    guint64 i, j;
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
            if (matrix[j][i])
                array[j] = matrix[j][i];
        /* change it to opposite will help with cache, but become confusing */
    }

    
    // clean up
    free_draw_dict(dd);
    if (cache->core->destroy){
        // if it is LRU, then it doesn't have a destroy function
        cache->core->destroy(cache);
    }
    else{
        cache_destroy(cache);
    }
    return ret_array;
}

static PyObject* differential_heatmap_with_Optimal(PyObject* self, PyObject* args, PyObject* keywds){
    PyObject* po;
    READER* reader;
    int num_of_threads = 4;
    long cache_size;
    char *name;
    char* plot_type_s;
    int plot_type;
    char* mode;
    long time_interval;
    struct_cache* cache;

    
    static char *kwlist[] = {"reader", "cache_size", "cache_type", "mode", "time_interval", "plot_type", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Olssls|$i", kwlist, &po, &cache_size, &name, &mode, &time_interval, &plot_type_s, &num_of_threads)) {
        printf("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    
    printf("plot type: %s, cache size: %ld, mode: %s, time_interval: %ld, num_of_threads: %d\n", plot_type_s, cache_size, mode, time_interval, num_of_threads);
    
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache
    char data_type = reader->type;
    
    if (strcmp(name, "FIFO") == 0){
        cache = fifo_init(cache_size, data_type, NULL);
    }
    
    else if (strcmp(name, "Optimal") == 0){
        struct optimal_init_params init_params = {.reader=reader, .next_access=NULL};
        cache = optimal_init(cache_size, data_type, (void*)&init_params);
    }
    else if (strcmp(name, "LRU") == 0){
        cache = cache_init(cache_size, data_type);
        cache->core->type = e_LRU;
    }
    else {
        printf("does not support given cache replacement algorithm: %s\n", name);
        exit(1);
    }
    
    if (strcmp(plot_type_s, "hit_rate_start_time_end_time") == 0)
        plot_type = hit_rate_start_time_end_time;
    else if (strcmp(plot_type_s, "hit_rate_start_time_cache_size") == 0)
        plot_type = hit_rate_start_time_cache_size;
    else if (strcmp(plot_type_s, "avg_rd_start_time_end_time") == 0)
        plot_type = avg_rd_start_time_end_time;
    else if (strcmp(plot_type_s, "cold_miss_count_start_time_end_time") == 0)
        plot_type = cold_miss_count_start_time_end_time;
    else {
        printf("unsupported plot type\n");
        exit(1);
    }
    
    struct optimal_init_params init_params = {.reader=reader, .next_access=NULL, .ts=0};
    struct_cache* optimal = optimal_init(cache_size, data_type, (void*)&init_params);
    

    
    printf("before computation\n");
    draw_dict* dd = differential_heatmap(reader, cache, optimal, *mode, time_interval, plot_type, num_of_threads);
    //    draw_dict* dd = heatmap(reader, cache, *mode, time_interval, plot_type, num_of_threads);
    printf("after computation\n");
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
    
    guint64 i, j;
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
//            if (matrix[j][i])
                array[j] = matrix[j][i];
            
    }
    
    
    // clean up
    free_draw_dict(dd);
    if (cache->core->destroy){
        // if it is LRU, then it doesn't have a destroy function
        cache->core->destroy(cache);
    }
    else{
        cache_destroy(cache);
    }
    optimal->core->destroy(optimal);
    return ret_array;
}


static PyObject* differential_heatmap_py(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    READER* reader;
    int num_of_threads = 4;
    long cache_size;
    char *name[2];
    char* plot_type_s;
    int plot_type;
    char* mode;
    long time_interval;
    struct_cache* cache[2];
    
    static char *kwlist[] = {"reader", "cache_size", "cache_type1", "cache_type2", "mode", "time_interval", "plot_type", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Olsssls|$i", kwlist, &po, &cache_size, &name[0], &name[1], &mode, &time_interval, &plot_type_s, &num_of_threads)) {
        printf("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    
    
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache
    char data_type = reader->type;
    
    guint64 i, j;
    for (i=0; i<2; i++){
        if (strcmp(name[i], "FIFO") == 0){
            cache[i] = fifo_init(cache_size, data_type, NULL);
        }
    
        else if (strcmp(name[i], "Optimal") == 0){
            struct optimal_init_params init_params = {.reader=reader, .next_access=NULL};
            cache[i] = optimal_init(cache_size, data_type, (void*)&init_params);
        }
        else if (strcmp(name[i], "LRU") == 0){
            cache[i] = cache_init(cache_size, data_type);
            cache[i]->core->type = e_LRU;
        }
        else {
            printf("does not support given cache replacement algorithm: %s\n", name[i]);
            exit(1);
        }
    }
    
    if (strcmp(plot_type_s, "hit_rate_start_time_end_time") == 0)
        plot_type = hit_rate_start_time_end_time;
    else if (strcmp(plot_type_s, "hit_rate_start_time_cache_size") == 0)
        plot_type = hit_rate_start_time_cache_size;
    else if (strcmp(plot_type_s, "avg_rd_start_time_end_time") == 0)
        plot_type = avg_rd_start_time_end_time;
    else if (strcmp(plot_type_s, "cold_miss_count_start_time_end_time") == 0)
        plot_type = cold_miss_count_start_time_end_time;
    else {
        printf("unsupported plot type\n");
        exit(1);
    }
    
    DEBUG(printf("before computation\n"));
    draw_dict* dd = differential_heatmap(reader, cache[0], cache[1], *mode, time_interval, plot_type, num_of_threads);
    DEBUG(printf("after computation\n"));
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
    
    //    PyObject* ret_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
//            if (matrix[j][i])
                array[j] = matrix[j][i];
    }
    
    
    // clean up
    free_draw_dict(dd);
    for (i=0; i<2; i++){
        if (cache[i]->core->destroy){
            // if it is LRU, then it doesn't have a destroy function
            cache[i]->core->destroy(cache[i]);
        }
        else{
            cache_destroy(cache[i]);
        }
    }
    return ret_array;
}


static PyObject* heatmap_rd_distribution_py(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    READER* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osl|$i", kwlist, &po, &mode, &time_interval, &num_of_threads)) {
        printf("parsing argument failed in heatmap_rd_distribution\n");
        return NULL;
    }
    
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    
    printf("before computation\n");
    draw_dict* dd = heatmap(reader, NULL, *mode, time_interval, rd_distribution, num_of_threads); 
    printf("after computation\n");
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_LONGLONG, 0);
    
    
    guint64 i, j;
    double **matrix = dd->matrix;
    long long *array;
    for (i=0; i<dd->ylength; i++){
        array = (long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
                array[j] = (long long)matrix[j][i];
    }
    printf("done copy\n");
    
    // clean up
    free_draw_dict(dd);
//    return ret_array;
    return Py_BuildValue("Nf", ret_array, reader->log_base);
}


static PyObject* heatmap_future_rd_distribution_py(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* po;
    READER* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osl|$i", kwlist, &po, &mode, &time_interval, &num_of_threads)) {
        printf("parsing argument failed in heatmap_rd_distribution\n");
        return NULL;
    }
    
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    
    printf("before computation\n");
    draw_dict* dd = heatmap(reader, NULL, *mode, time_interval, future_rd_distribution, num_of_threads);
    printf("after computation\n");
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_LONGLONG, 0);
    
    
    guint64 i, j;
    double **matrix = dd->matrix;
    long long *array;
    for (i=0; i<dd->ylength; i++){
        array = (long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
            array[j] = (long long)matrix[j][i];
    }
    printf("done copy\n");
    
    // clean up
    free_draw_dict(dd);
    //    return ret_array;
    return Py_BuildValue("Nf", ret_array, reader->log_base);
}


static PyMethodDef c_heatmap_funcs[] = {
    {"get_last_access_dist", (PyCFunction)heatmap_get_last_access_dist_seq,
        METH_VARARGS | METH_KEYWORDS, "get the distance to the last access time for each \
        request in the form of numpy array, -1 if haven't seen before"},
    {"get_next_access_dist", (PyCFunction)heatmap_get_next_access_dist_seq,
        METH_VARARGS | METH_KEYWORDS, "get the distance to the next access time for each \
        request in the form of numpy array, -1 if it won't be accessed any more"},
    {"heatmap", (PyCFunction)heatmap_computation,
        METH_VARARGS | METH_KEYWORDS, "heatmap pixel computation"},
    {"differential_heatmap_with_Optimal", (PyCFunction)differential_heatmap_with_Optimal,
        METH_VARARGS | METH_KEYWORDS, "differential heatmap pixel computation compared with Optimal"},
    {"differential_heatmap", (PyCFunction)differential_heatmap_py,
        METH_VARARGS | METH_KEYWORDS, "differential heatmap pixel computation"},
    {"heatmap_rd_distribution", (PyCFunction)heatmap_rd_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {"heatmap_future_rd_distribution", (PyCFunction)heatmap_future_rd_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef c_heatmap_definition = { 
    PyModuleDef_HEAD_INIT,
    "c_heatmap",
    "A Python module that doing heatmap related computation",
    -1, 
    c_heatmap_funcs
};



PyMODINIT_FUNC PyInit_c_heatmap(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&c_heatmap_definition);
}

