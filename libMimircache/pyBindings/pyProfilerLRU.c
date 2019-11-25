//
//  pyProfilerLRU.c
//  pyMimircache
//
//  Created by Juncheng on 5/26/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//


#include "python_wrapper.h"


static PyObject *ProfilerLRU_get_reuse_dist(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  static char *kwlist[] = {"reader", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &po)) {
    return NULL;
  }

  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  // get reuse dist
  gint64 *reuse_dist = _get_reuse_dist_seq(reader);

  // create numpy array
  npy_intp dims[1] = {get_num_of_req(reader)};
  PyObject *ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
  for (guint64 i = 0; i < get_num_of_req(reader); i++)
    *((long long *) PyArray_GETPTR1((PyArrayObject *) ret_array, i)) = (long long) reuse_dist[i];

  g_free(reuse_dist);
  return ret_array;
}


static PyObject *ProfilerLRU_get_future_reuse_dist(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  static char *kwlist[] = {"reader", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|ll", kwlist, &po)) {
    return NULL;
  }

  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  // get reuse dist
  gint64 *reuse_dist = get_future_reuse_dist(reader);

  // create numpy array
  npy_intp dims[1] = {get_num_of_req(reader)};
  PyObject *ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
  for (guint64 i = 0; i < get_num_of_req(reader); i++)
    *((long long *) PyArray_GETPTR1((PyArrayObject *) ret_array, i)) = (long long) reuse_dist[i];

  g_free(reuse_dist);
  return ret_array;
}


static PyObject *ProfilerLRU_get_miss_count(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  gint64 cache_size = -1;
  static char *kwlist[] = {"reader", "cache_size", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|l", kwlist,
                                   &po, &cache_size)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  // get miss count
  guint64 *miss_count = get_miss_count_seq(reader, cache_size);

  // create numpy array
  if (cache_size == -1) {
    cache_size = reader->base->n_total_req;
  }

  npy_intp dims[1] = { cache_size + 1 };
  PyObject *ret_array = PyArray_SimpleNew(1, dims, NPY_LONGLONG);
  for (gint64 i = 0; i < cache_size + 1; i++) {
    *((long long *) PyArray_GETPTR1((PyArrayObject *) ret_array, i)) = (long long) miss_count[i];
  }

  g_free(miss_count);
  return ret_array;
}


static PyObject *ProfilerLRU_get_miss_ratio(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  reader_t *reader;
  gint64 cache_size = -1;
  static char *kwlist[] = {"reader", "cache_size", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|l", kwlist,
                                   &po, &cache_size)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  // get miss count
  double *miss_ratio = get_miss_ratio_seq(reader, cache_size);

  // create numpy array
  if (cache_size == -1) {
    cache_size = reader->base->n_total_req;
  }

  npy_intp dims[1] = { cache_size + 1 };
  PyObject *ret_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  for (gint64 i = 0; i < cache_size + 1; i++) {
    *((double *) PyArray_GETPTR1((PyArrayObject *) ret_array, i)) = miss_ratio[i];
  }

  g_free(miss_ratio);
  return ret_array;
}


static PyMethodDef ProfilerLRU_funcs[] = {
  {"get_miss_count",        (PyCFunction) ProfilerLRU_get_miss_count,
    METH_VARARGS | METH_KEYWORDS, "get miss count as a numpy array"},
  {"get_hit_ratio_seq",     (PyCFunction) ProfilerLRU_get_miss_ratio,
    METH_VARARGS | METH_KEYWORDS, "get miss ratio as a numpy array"},
  {"get_future_reuse_dist", (PyCFunction) ProfilerLRU_get_future_reuse_dist,
    METH_VARARGS | METH_KEYWORDS, "get future reuse distance/stack distance as a numpy array"},
  {"get_reuse_dist",        (PyCFunction) ProfilerLRU_get_reuse_dist,
    METH_VARARGS | METH_KEYWORDS, "get reuse distance as a numpy array"},
//    {"save_reuse_dist", (PyCFunction)ProfilerLRU_save_reuse_dist,
//        METH_VARARGS | METH_KEYWORDS, "save reuse distance array to specified file"},
//    {"load_reuse_dist", (PyCFunction)ProfilerLRU_load_reuse_dist,
//        METH_VARARGS | METH_KEYWORDS, "load reuse distance array from specified file"},
  {NULL, NULL, 0,                 NULL}
};


static struct PyModuleDef ProfilerLRU_definition = {
  PyModuleDef_HEAD_INIT,
  "ProfilerLRU",
  "A Python module that doing profiling for LRU cache",
  -1,
  ProfilerLRU_funcs
};


PyMODINIT_FUNC PyInit_ProfilerLRU(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&ProfilerLRU_definition);
}

