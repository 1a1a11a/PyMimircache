//
//  pyUtils.c
//  PyMimircache
//
//  Created by Juncheng on 5/24/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//



#include <LRU_K.h>
#include "pyHeaders.h"
#include "../libMimircache/libMimircache/include/mimircache/plugin.h"

static void py_close_reader(PyObject *pycap_reader) {
  reader_t *reader = (reader_t *) PyCapsule_GetPointer(pycap_reader, NULL);
  DEBUG("reader (%s) is closed\n", reader->base->trace_path);
  close_reader(reader);
}

static void py_free_cache(PyObject *pycap_cache) {
  cache_t *cache = (cache_t *) PyCapsule_GetPointer(pycap_cache, NULL);
  DEBUG("free cache %s\n", cache->core->cache_name);
  cache->core->destroy(cache);
}

static PyObject* py_reset_reader(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  reader_t *reader;

  // parse arguments
  static char *kwlist[] = {"reader", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &po)) {
    ERROR("parsing argument failed in %s\n", __func__);
    PyErr_SetString(PyExc_RuntimeError, "parsing argument failed");
    Py_RETURN_NONE;
  }

  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    ERROR("cannot get reader from capsule");
    PyErr_SetString(PyExc_RuntimeError, "cannot get reader from capsule");
    Py_RETURN_NONE;
  }

  reset_reader(reader);
}


static PyObject* py_setup_cache(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *po;
  PyObject *cache_params = NULL;
  reader_t *reader;
  long cache_size;
  char *algorithm;
  char *obj_id_type_str = "c";
  obj_id_t obj_id_type;
  cache_t *cache;

  static char *kwlist[] = {"algorithm", "cache_size", "obj_id_type", "cache_params", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "sls|$O", kwlist,
                                   &algorithm, &cache_size, &obj_id_type_str, &cache_params)) {
    ERROR("parsing argument failed in %s\n", __func__);
    PyErr_SetString(PyExc_RuntimeError, "parsing argument failed");
    Py_RETURN_NONE;
  }

  if (obj_id_type_str[0] == 'c')
    obj_id_type = OBJ_ID_STR;
  else if (obj_id_type_str[0] == 'l')
    obj_id_type = OBJ_ID_NUM;
  else {
    PyErr_SetString(PyExc_RuntimeError, "unknown obj_id type, supported obj_id types are c,l\n");
    Py_RETURN_NONE;
  }

//  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
//    ERROR("error retrieval reader from capsule in %s\n", __func__);
//    PyErr_SetString(PyExc_RuntimeError, "error retrieval reader from capsule");
//    Py_RETURN_NONE;
//  }

  // build cache
  cache = py_create_cache(algorithm, cache_size, obj_id_type, cache_params, 0);
  return PyCapsule_New((void *) cache, NULL, py_free_cache);
}

static PyObject *py_setup_reader(PyObject *self, PyObject *args, PyObject *keywds) {
  char *trace_path;
  char *trace_type_str;
  trace_type_t trace_type;
  char *obj_id_type_str = "c";
  obj_id_t obj_id_type;
  PyObject *py_init_params;
  void *init_params = NULL;

  static char *kwlist[] = {"trace_path", "trace_type", "obj_id_type", "init_params", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "sss|O", kwlist, &trace_path,
                                   &trace_type_str, &obj_id_type_str, &py_init_params)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "parsing argument failed in setup reader\n");
    Py_RETURN_NONE;
  }

  if (obj_id_type_str[0] == 'c')
    obj_id_type = OBJ_ID_STR;
  else if (obj_id_type_str[0] == 'l')
    obj_id_type = OBJ_ID_NUM;
  else {
    PyErr_SetString(PyExc_RuntimeError, "unknown obj_id type, supported obj_id types are c,l\n");
    Py_RETURN_NONE;
  }
  if (trace_type_str[0] == 'c') {
    trace_type = CSV_TRACE;
    if (!PyDict_Check(py_init_params)) {
      PyErr_SetString(PyExc_RuntimeError, "init_params is not a valid python dictionary\n");
      Py_RETURN_NONE;
    }
    init_params = (void *) new_csvReader_init_params(-1, -1, -1, -1, FALSE, ',', -1);
    /* if it is csv file, we need extra init parameters */
    PyObject *py_obj_id, *py_size, *py_op, *py_real_time,
      *py_header, *py_delimiter, *py_traceID;
    csvReader_init_params *csv_init_params = init_params;

    if ((py_obj_id = PyDict_GetItemString(py_init_params, "obj_id_field")) != NULL) {
      csv_init_params->obj_id_field = (gint) PyLong_AsLong(py_obj_id);
    }

    if ((py_size = PyDict_GetItemString(py_init_params, "obj_size_field")) != NULL) {
      csv_init_params->size_field = (gint) PyLong_AsLong(py_size);
    }

    if ((py_op = PyDict_GetItemString(py_init_params, "op_field")) != NULL) {
      csv_init_params->op_field = (gint) PyLong_AsLong(py_op);
    }

    if ((py_real_time = PyDict_GetItemString(py_init_params, "real_time_field")) != NULL) {
      csv_init_params->real_time_field = (gint) PyLong_AsLong(py_real_time);
    }

    if ((py_header = PyDict_GetItemString(py_init_params, "header")) != NULL) {
      csv_init_params->has_header = PyObject_IsTrue(py_header);
    }

    if ((py_delimiter = PyDict_GetItemString(py_init_params, "delimiter")) != NULL) {
      csv_init_params->delimiter = *((unsigned char *) PyUnicode_AsUTF8(py_delimiter));
    }

    if ((py_traceID = PyDict_GetItemString(py_init_params, "traceID")) != NULL) {
      csv_init_params->traceID_field = (gint) PyLong_AsLong(py_traceID);
    }

    if (((csvReader_init_params *) init_params)->has_header) DEBUG("csv data has header");

    DEBUG2("delimiter %d(%c)\n", ((csvReader_init_params *) init_params)->delimiter,
          ((csvReader_init_params *) init_params)->delimiter);
  } else if (trace_type_str[0] == 'b') {
    trace_type = BIN_TRACE;
    if (!PyDict_Check(py_init_params)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "input init_params is not a valid python dictionary\n");
      Py_RETURN_NONE;
    }

    init_params = g_new0(binary_init_params_t, 1);
    binary_init_params_t *bin_init_params = init_params;
    PyObject *py_obj_id, *py_size, *py_op, *py_real_time, *py_fmt;

    if ((py_obj_id = PyDict_GetItemString(py_init_params, "obj_id_field")) != NULL) {
      bin_init_params->obj_id_pos = (gint) PyLong_AsLong(py_obj_id);
    }

    if ((py_size = PyDict_GetItemString(py_init_params, "size")) != NULL) {
      bin_init_params->size_pos = (gint) PyLong_AsLong(py_size);
    }

    if ((py_op = PyDict_GetItemString(py_init_params, "op")) != NULL) {
      bin_init_params->op_pos = (gint) PyLong_AsLong(py_op);
    }

    if ((py_real_time = PyDict_GetItemString(py_init_params, "real_time")) != NULL) {
      bin_init_params->real_time_pos = (gint) PyLong_AsLong(py_real_time);
    }

    py_fmt = PyDict_GetItemString(py_init_params, "fmt");
    if (!PyUnicode_Check(py_fmt)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "passed format string is not unicode \n");
      Py_RETURN_NONE;
    }
    if (PyUnicode_READY(py_fmt) != 0) {
      PyErr_SetString(PyExc_RuntimeError,
                      "failed get fmt unicode ready\n");
      Py_RETURN_NONE;
    }

    Py_UCS1 *py_ucs1 = PyUnicode_1BYTE_DATA(py_fmt);

    DEBUG2("binary fmt %s\n", py_ucs1);
    strcpy(bin_init_params->fmt, (char *) py_ucs1);

  } else if (trace_type_str[0] == 'p') {
    trace_type = PLAIN_TXT_TRACE;
  } else if (trace_type_str[0] == 'v') {
    trace_type = VSCSI_TRACE;
    obj_id_type = OBJ_ID_NUM;
  } else {
    PyErr_SetString(PyExc_RuntimeError, "unknown trace type, supported trace types are c,p,b,v\n");
    Py_RETURN_NONE;
  }

  reader_t *reader = setup_reader(trace_path, trace_type, obj_id_type, init_params);

  if (init_params != NULL) {
    g_free(init_params);
  }

  return PyCapsule_New((void *) reader, NULL, py_close_reader);
//  SUPPRESS_FUNCTION_NO_USE_WARNING(py_close_reader);
//    return PyCapsule_New((void *)reader, NULL, reader_pycapsule_destructor);
}


static PyMethodDef PyUtils_funcs[] = {
  {"setup_cache",  (PyCFunction) py_setup_cache,
    METH_VARARGS | METH_KEYWORDS, "setup a cache in libMimircache"},
  {"setup_reader", (PyCFunction) py_setup_reader,
    METH_VARARGS | METH_KEYWORDS, "setup a reader in libMimircache"},
  {"reset_reader", (PyCFunction) py_reset_reader,
    METH_VARARGS | METH_KEYWORDS, "reset the reader in libMimircache"},
  {NULL, NULL, 0,                 NULL}
};


static struct PyModuleDef PyUtils_definition = {
  PyModuleDef_HEAD_INIT,
  "PyUtils",
  "A Python module that setup cache and reader in libMimircache",
  -1,
  PyUtils_funcs
};


PyMODINIT_FUNC PyInit_PyUtils(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&PyUtils_definition);
}
