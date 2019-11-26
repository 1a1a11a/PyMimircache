//
//  pyProfiler.c
//  PyMimircache
//
//  Created by Juncheng on 15/26/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//


#include "pyHeaders.h"


static PyObject *Profiler_run_trace(PyObject *self,
                                    PyObject *args,
                                    PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  int num_of_threads = 4;
  long cache_size;
  int bin_size = -1;
  char *algorithm;
  cache_t *cache;
  PyObject *cache_params;

  static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
                           "cache_params", "num_of_threads", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oi", kwlist, &po,
                                   &algorithm, &cache_size, &bin_size,
                                   &cache_params, &num_of_threads)) {
    ERROR("parsing argument failed");
    PyErr_SetString(PyExc_RuntimeError, "parsing argument failed");
    return NULL;
  }

  DEBUG("bin size: %d, threads: %d\n", bin_size, num_of_threads);

  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    ERROR("reader pointer error");
    PyErr_SetString(PyExc_RuntimeError, "reader pointer error");
    return NULL;
  }

  // build cache
  cache = py_create_cache(algorithm, cache_size, reader->base->obj_id_type, cache_params, 0);

  // run the trace
  profiler_res_t **results = run_trace(reader, cache, num_of_threads, bin_size);

  // create 5-d numpy array, the first dim is cache size, the second is obj miss cnt,
  // the third dim is obj req cnt, the fourth is miss byte, the fifth is req byte
  guint num_of_bins = ceil((double) cache_size / bin_size) + 1;
  npy_intp dims[2] = {5, num_of_bins};
  // return a
  PyObject *ret_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  guint i;
  double *arrays[5];
  arrays[0] = (double *) PyArray_GETPTR1((PyArrayObject *) ret_array, 0);
  arrays[1] = (double *) PyArray_GETPTR1((PyArrayObject *) ret_array, 1);
  arrays[2] = (double *) PyArray_GETPTR1((PyArrayObject *) ret_array, 2);
  arrays[3] = (double *) PyArray_GETPTR1((PyArrayObject *) ret_array, 3);
  arrays[4] = (double *) PyArray_GETPTR1((PyArrayObject *) ret_array, 4);
  for (i = 0; i < num_of_bins; i++) {
    arrays[0][i] = results[i]->cache_size;
    arrays[1][i] = results[i]->miss_cnt;
    arrays[2][i] = results[i]->req_cnt;
    arrays[3][i] = results[i]->miss_byte;
    arrays[4][i] = results[i]->req_byte;
    g_free(results[i]);
  }

  g_free(results);
  cache->core->destroy(cache);
  return ret_array;
}


static PyObject *Profiler_get_eviction_age(PyObject *self,
                                           PyObject *args,
                                           PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  int num_of_threads = 4;
  long cache_size;
  int bin_size;
  char *algorithm;
  cache_t *cache;
  PyObject *cache_params;
  gint64 i, j;

  static char *kwlist[] = {"reader", "algorithm", "cache_size", "bin_size",
                           "cache_params", "num_of_threads", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Osli|Oi", kwlist, &po,
                                   &algorithm, &cache_size, &bin_size,
                                   &cache_params, &num_of_threads)) {
    ERROR("parsing argument failed");
    return NULL;
  }


  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    ERROR("reader pointer error");
    return NULL;
  }

  // build cache
  cache = py_create_cache(algorithm, cache_size, reader->base->obj_id_type, cache_params, 0);

  profiler_res_t **results = run_trace(reader, cache, num_of_threads, bin_size);

  // create numpy array
  guint num_of_bins = ceil((double) cache_size / bin_size) + 1;
  npy_intp dims[2] = {num_of_bins, reader->base->n_total_req};
  PyObject *ret_array = PyArray_SimpleNew(2, dims, NPY_LONG);

  long *one_dim_array;
  for (i = 0; i < num_of_bins; i++) {
    one_dim_array = PyArray_GETPTR1((PyArrayObject *) ret_array, i);
    if (i == 0)
      memset(one_dim_array, 0, sizeof(long) * reader->base->n_total_req);
    else {
      for (j = 0; j < reader->base->n_total_req; j++) {
        one_dim_array[j] = ((gint64 *) results[i]->other_data)[j];
      }
    }


    g_free(results[i]->other_data);
    g_free(results[i]);
  }

  g_free(results);
  cache->core->destroy(cache);
  return ret_array;
}


static PyMethodDef Profiler_funcs[] = {
  {"run_trace",        (PyCFunction) Profiler_run_trace,
    METH_VARARGS | METH_KEYWORDS, "run the given trace and obtain hit/miss ratio/count at different cache sizes"},
  {"get_eviction_age", (PyCFunction) Profiler_get_eviction_age,
    METH_VARARGS | METH_KEYWORDS, "get eviction age as a numpy array"},
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef Profiler_definition = {
  PyModuleDef_HEAD_INIT,
  "Profiler",
  "A Python module in C that performs simulations",
  -1,
  Profiler_funcs
};


PyMODINIT_FUNC PyInit_Profiler(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&Profiler_definition);
}

