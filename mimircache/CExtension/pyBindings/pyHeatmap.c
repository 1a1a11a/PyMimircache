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
#include "python_wrapper.h"


#define NPY_NO_DEPRECATED_API 11
#include <numpy/arrayobject.h>


/* TODO: 
not urgent, not necessary: change this profiler module into a python object,
    this is not necessary for now because we are not going to expose this level API 
    to user, instead we wrap it with our python API, so these C functions are only 
    called inside mimircache  
*/


static PyObject* differential_heatmap_py(PyObject* self,
                                         PyObject* args,
                                         PyObject* keywds);




static PyObject* heatmap_get_last_access_dist_seq(PyObject* self,
                                                  PyObject* args,
                                                  PyObject* keywds)
{   
    PyObject* po;
    reader_t* reader;
    long begin=-1, end=-1; 
    static char *kwlist[] = {"reader", "begin", "end", NULL};

    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist, 
                                &po, &begin, &end)) {
        // currently specifying begin and ending position is not supported 
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // get last access dist list  
    GSList* list = get_last_access_dist_seq(reader, read_one_element);

    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);
    
    if (begin == -1)
        begin = 0;
    if (end == -1)
        end = reader->base->total_num;

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


static PyObject* heatmap_get_next_access_dist_seq(PyObject* self,
                                                  PyObject* args,
                                                  PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    long begin=-1, end=-1;
    static char *kwlist[] = {"reader", "begin", "end", NULL};
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist,
                                     &po, &begin, &end)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }

    // get reversed last access dist list
    GSList* list = get_last_access_dist_seq(reader, read_one_element_above);

    if (reader->base->total_num == -1)
        get_num_of_cache_lines(reader);

    if (begin == -1)
        begin = 0;
    if (end == -1)
        end = reader->base->total_num;

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


static PyObject* heatmap_computation(PyObject* self,
                                     PyObject* args,
                                     PyObject* keywds)
{
    PyObject* po;
    PyObject* cache_params=NULL;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    char *algorithm;
    char* plot_type_s;
    heatmap_type_e plot_type;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = 200;
    struct_cache* cache;

    static char *kwlist[] = {"reader", "mode", "plot_type", "cache_size",
        "algorithm", "time_interval", "num_of_pixels", "cache_params", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Ossls|$llOi", kwlist, &po,
                                     &mode, &plot_type_s, &cache_size, &algorithm,
                                     &time_interval, &num_of_pixels, &cache_params,
                                     &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;

    
    INFO("plot type: %s, cache size: %ld, mode: %s, time_interval: %ld, "
           "num_of_pixels: %ld, num_of_threads: %d\n", plot_type_s, cache_size,
           mode, time_interval, num_of_pixels, num_of_threads);
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache
    if (strcmp(algorithm, "LRU") == 0){
        cache = cache_init(cache_size, reader->base->type, 0);
        cache->core->type = e_LRU;
    }
    else
        cache = build_cache(reader, cache_size, algorithm, cache_params, 0);

    
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
    
    draw_dict* dd = heatmap(reader, cache, *mode, time_interval, num_of_pixels,
                            plot_type, num_of_threads);
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };

    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);

    
    guint64 i, j;
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
            if (matrix[j][i]){
                array[j] = matrix[j][i];
//                printf("%lu %lu: %lf\n", i, j, matrix[i][j]);
            }
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

static PyObject* differential_heatmap_with_Optimal(PyObject* self,
                                                   PyObject* args,
                                                   PyObject* keywds){
    PyObject* po;
    PyObject* cache_params;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    char *algorithm;
    char* plot_type_s;
    int plot_type;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = 200;
    struct_cache* cache;

    
    static char *kwlist[] = {"reader", "mode", "plot_type", "cache_size",
        "algorithm", "time_interval", "num_of_pixels", "cache_params",
        "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Ossls|$llOi", kwlist, &po,
                                     &mode, &plot_type_s, &cache_size, &algorithm,
                                     &time_interval, &num_of_pixels, &cache_params,
                                     &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;
    
    
    printf("plot type: %s, cache size: %ld, mode: %s, time_interval: %ld, "
           "num_of_threads: %d\n", plot_type_s, cache_size, mode, time_interval,
           num_of_threads);
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache
    if (strcmp(algorithm, "LRU") == 0){
        cache = cache_init(cache_size, reader->base->type, 0);
        cache->core->type = e_LRU;
    }
    else
        cache = build_cache(reader, cache_size, algorithm, cache_params, 0);

    
    if (strcmp(plot_type_s, "hit_rate_start_time_end_time") == 0)
        plot_type = hit_rate_start_time_end_time;
    else if (strcmp(plot_type_s, "hit_rate_start_time_cache_size") == 0)
        plot_type = hit_rate_start_time_cache_size;
    else if (strcmp(plot_type_s, "avg_rd_start_time_end_time") == 0)
        plot_type = avg_rd_start_time_end_time;
    else if (strcmp(plot_type_s, "cold_miss_count_start_time_end_time") == 0)
        plot_type = cold_miss_count_start_time_end_time;
    else {
        ERROR("unsupported plot type\n");
        exit(1);
    }
    
    struct optimal_init_params init_params = {.reader=reader, .next_access=NULL, .ts=0};
    struct_cache* optimal = optimal_init(cache_size, reader->base->type, 0, (void*)&init_params);
    

    
    draw_dict* dd = differential_heatmap(reader, cache, optimal, *mode,
                                         time_interval, num_of_pixels,
                                         plot_type, num_of_threads);
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
    
    guint64 i, j;
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
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


static PyObject* differential_heatmap_py(PyObject* self,
                                         PyObject* args,
                                         PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    long cache_size;
    char *algorithm[2];
    char* plot_type_s;
    int plot_type;
    char* mode;
    PyObject* cache_params[2];
    long time_interval = -1;
    long num_of_pixels = 200;
    struct_cache* cache[2];
    
    static char *kwlist[] = {"reader", "mode", "plot_type", "cache_size",
        "algorithm1", "algorithm2", "time_interval", "num_of_pixels",
        "cache_params1", "cache_params2", "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osslss|$llOOi", kwlist, &po,
                                     &mode, &plot_type_s, &cache_size,
                                     &algorithm[0], &algorithm[1], &time_interval,
                                     &num_of_pixels, &(cache_params[0]),
                                     &(cache_params[1]), &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_computation\n");
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;

    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    // build cache (isolate LRU, because we don't need a real LRU for profiling,
    // just need to pass size and data_type)
    guint64 i, j;
    for (i=0; i<2; i++){
        if (strcmp(algorithm[i], "LRU") == 0){
            char data_type;
            if (reader->base->type == 'v')
                data_type= 'l';
            else
                data_type = 'c';
            cache[i] = cache_init(cache_size, data_type, 0);
            cache[i]->core->type = e_LRU;
        }
        else
            cache[i] = build_cache(reader, cache_size, algorithm[i], cache_params[i], 0);
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
        ERROR("unsupported plot type\n");
        exit(1);
    }
    
    draw_dict* dd = differential_heatmap(reader, cache[0], cache[1], *mode,
                                         time_interval, num_of_pixels,
                                         plot_type, num_of_threads);
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
    
    //    PyObject* ret_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double **matrix = dd->matrix;
    double *array;
    for (i=0; i<dd->ylength; i++){
        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
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


static PyObject* heatmap_rd_distribution_py(PyObject* self,
                                            PyObject* args,
                                            PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = -1;
    int CDF = 0;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_pixels",
        "num_of_threads", "CDF", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Os|$llii", kwlist, &po,
                                     &mode, &time_interval, &num_of_pixels,
                                     &num_of_threads, &CDF)) {
        ERROR("parsing argument failed in heatmap_rd_distribution\n");
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;

    
    draw_dict* dd;
    if (CDF){
        dd = heatmap(reader, NULL, *mode, time_interval, num_of_pixels,
                     rd_distribution_CDF, num_of_threads);
    }
    else
        dd = heatmap(reader, NULL, *mode, time_interval, num_of_pixels,
                     rd_distribution, num_of_threads);
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array;
    guint64 i, j;
//    long long **matrix = dd->matrix;
    double **matrix = dd->matrix;

    
    if (!CDF){
        ret_array = PyArray_EMPTY(2, dims, NPY_LONGLONG, 0);
//        ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
//        double *array;
//        gint64 *sum_array = g_new0(gint64, dd->xlength);
//        for (i=0; i<dd->xlength; i++)
//            for (j=0; j<dd->ylength; j++)
//                sum_array[i] += (long long)matrix[i][j];
//        for (i=0; i<dd->ylength; i++){
//            array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
//            for (j=0; j<dd->xlength; j++)
//                array[j] = (double)matrix[j][i] / sum_array[j];
        
        
        long long *array;
        for (i=0; i<dd->ylength; i++){
            array = (long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
            for (j=0; j<dd->xlength; j++){
                array[j] = (long long)matrix[j][i];
            }
        }
    }
    else{
        ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
        double *array;
        for (i=0; i<dd->ylength; i++){
            array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
            for (j=0; j<dd->xlength; j++){
                array[j] = matrix[j][i];
            }
        }

    }
        
    
    // clean up
    free_draw_dict(dd);
    return Py_BuildValue("Nf", ret_array, reader->udata->log_base);
}


static PyObject* heatmap_future_rd_distribution_py(PyObject* self,
                                                   PyObject* args,
                                                   PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = 200;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_pixels",
        "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Os|$lli", kwlist, &po, &mode,
                                     &time_interval, &num_of_pixels, &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_rd_distribution\n");
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;

    
    draw_dict* dd = heatmap(reader, NULL, *mode, time_interval, num_of_pixels,
                            future_rd_distribution, num_of_threads);
    
    // create numpy array
    npy_intp dims[2] = { dd->ylength, dd->xlength };
    
    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_LONGLONG, 0);
    
    
    guint64 i, j;
//    long long **matrix = dd->matrix;
    double **matrix = dd->matrix;
    long long *array;
    for (i=0; i<dd->ylength; i++){
        array = (long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
        for (j=0; j<dd->xlength; j++)
            array[j] = (long long)matrix[j][i];
    }
    
    
//    PyObject* ret_array = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    
//    double *array;
//    gint64 *sum_array = g_new0(gint64, dd->xlength);
//    for (i=0; i<dd->xlength; i++)
//        for (j=0; j<dd->ylength; j++)
//            sum_array[i] += (long long)matrix[i][j];
//    for (i=0; i<dd->ylength; i++){
//        array = (double*) PyArray_GETPTR1((PyArrayObject *)ret_array, i);
//        for (j=0; j<dd->xlength; j++)
//            array[j] = (double)matrix[j][i] / sum_array[j];
//    }
    
    
    // clean up
    free_draw_dict(dd);
    return Py_BuildValue("Nf", ret_array, reader->udata->log_base);
}


static PyObject* heatmap_dist_distribution_py(PyObject* self,
                                                   PyObject* args,
                                                   PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = 200;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_pixels",
        "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Os|$lli", kwlist, &po, &mode,
                                     &time_interval, &num_of_pixels, &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_dist_distribution\n");
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;
    
    
    draw_dict* dd = heatmap(reader, NULL, *mode, time_interval, num_of_pixels,
                            dist_distribution, num_of_threads);
    
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
    
    
    // clean up
    free_draw_dict(dd);
    return Py_BuildValue("Nf", ret_array, reader->udata->log_base);
}


static PyObject* heatmap_rt_distribution_py(PyObject* self,
                                              PyObject* args,
                                              PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    int num_of_threads = 4;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = 200;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_pixels",
        "num_of_threads", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Os|$lli", kwlist, &po, &mode,
                                     &time_interval, &num_of_pixels, &num_of_threads)) {
        ERROR("parsing argument failed in heatmap_dist_distribution\n");
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;
    
    
    draw_dict* dd = heatmap(reader, NULL, *mode, time_interval, num_of_pixels,
                            rt_distribution, num_of_threads);
    
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
    
    
    // clean up
    free_draw_dict(dd);
    return Py_BuildValue("Nf", ret_array, reader->udata->log_base);
}


static PyObject* heatmap_get_break_points(PyObject* self,
                                          PyObject* args,
                                          PyObject* keywds)
{
    PyObject* po;
    reader_t* reader;
    char* mode;
    long time_interval = -1;
    long num_of_pixels = -1;
    
    static char *kwlist[] = {"reader", "mode", "time_interval", "num_of_pixels", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Os|ll", kwlist, &po, &mode,
                                     &time_interval, &num_of_pixels)) {
        ERROR("parsing argument failed in heatmap_get_break_points\n");
        return NULL;
    }
    
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    if (time_interval == -1 && num_of_pixels == -1)
        num_of_pixels = 200;
    
    GArray* breakpoints;
    if (mode[0] == 'r')
        breakpoints = gen_breakpoints_realtime(reader, (guint64)time_interval,
                                               num_of_pixels);
    else
        breakpoints = gen_breakpoints_virtualtime(reader, (guint64)time_interval,
                                                  num_of_pixels);
    
    // create numpy array
    npy_intp dims[1] = { breakpoints->len };
    
    PyObject* ret_array = PyArray_EMPTY(1, dims, NPY_LONGLONG, 0);
    
    
    guint64 i;
    for (i=0; i<breakpoints->len; i++){
        *(long long*) PyArray_GETPTR1((PyArrayObject *)ret_array, i) =
            (long long)g_array_index(breakpoints, guint64, i);
    }

    
    // clean up
//    g_array_free(breakpoints, TRUE);      DON'T FREE it, if you are going to free it, also remember to free break point struct
    return ret_array;
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
    {"diffHeatmap", (PyCFunction)differential_heatmap_py,
        METH_VARARGS | METH_KEYWORDS, "differential heatmap pixel computation"},
    {"heatmap_rd_distribution", (PyCFunction)heatmap_rd_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {"heatmap_future_rd_distribution", (PyCFunction)heatmap_future_rd_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {"heatmap_dist_distribution", (PyCFunction)heatmap_dist_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {"heatmap_reuse_time_distribution", (PyCFunction)heatmap_rt_distribution_py,
        METH_VARARGS | METH_KEYWORDS, "reuse distance distribution heatmap"},
    {"getBreakpoints", (PyCFunction)heatmap_get_break_points,
        METH_VARARGS | METH_KEYWORDS, "generate virtual/real break points"},
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

