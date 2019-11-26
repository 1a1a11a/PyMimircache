//
//  pyReader.c
//  PyMimircache
//
//  Created by Juncheng on 5/26/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//


#include "python_wrapper.c"


static void reader_pycapsule_destructor(PyObject *pycap_reader) {
  reader_t *reader = (reader_t *) PyCapsule_GetPointer(pycap_reader, NULL);
  close_reader(reader);
}

static PyObject *reader_setup_reader(PyObject *self, PyObject *args, PyObject *keywds) {
  char *file_loc;
  char *trace_type_str;
  trace_type_t trace_type;
  char *obj_id_type_str = "c";
  obj_id_t obj_id_type;
  PyObject *py_init_params;
  void *init_params = NULL;

  static char *kwlist[] = {"file_loc", "trace_type", "obj_id_type", "init_params", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|sO", kwlist, &file_loc,
                                   &trace_type_str, &obj_id_type_str, &py_init_params)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "parsing argument failed in setup reader\n");
    return NULL;
  }

  if (obj_id_type_str[0] == 'c')
    obj_id_type = OBJ_ID_STR;
  else if (obj_id_type_str[0] == 'l')
    obj_id_type = OBJ_ID_NUM;
  else {
    PyErr_SetString(PyExc_RuntimeError, "unknown obj_id type, supported obj_id types are c,l\n");
    return NULL;
  }
  if (trace_type_str[0] == 'c') {
    trace_type = CSV_TRACE;
    if (!PyDict_Check(py_init_params)) {
      PyErr_SetString(PyExc_RuntimeError, "init_params is not a valid python dictionary\n");
      return NULL;
    }
    init_params = (void *) new_csvReader_init_params(-1, -1, -1, -1, FALSE, ',', -1);
    /* if it is csv file, we need extra init parameters */
    PyObject *py_label, *py_size, *py_op, *py_real_time,
      *py_header, *py_delimiter, *py_traceID;
    csvReader_init_params *csv_init_params = init_params;

    if ((py_label = PyDict_GetItemString(py_init_params, "label")) != NULL) {
      csv_init_params->label_column = (gint) PyLong_AsLong(py_label);
    }

    if ((py_size = PyDict_GetItemString(py_init_params, "size")) != NULL) {
      csv_init_params->size_column = (gint) PyLong_AsLong(py_size);
    }

    if ((py_op = PyDict_GetItemString(py_init_params, "op")) != NULL) {
      csv_init_params->op_column = (gint) PyLong_AsLong(py_op);
    }

    if ((py_real_time = PyDict_GetItemString(py_init_params, "real_time")) != NULL) {
      csv_init_params->real_time_column = (gint) PyLong_AsLong(py_real_time);
    }

    if ((py_header = PyDict_GetItemString(py_init_params, "header")) != NULL) {
      csv_init_params->has_header = PyObject_IsTrue(py_header);
    }

    if ((py_delimiter = PyDict_GetItemString(py_init_params, "delimiter")) != NULL) {
      csv_init_params->delimiter = *((unsigned char *) PyUnicode_AsUTF8(py_delimiter));
    }

    if ((py_traceID = PyDict_GetItemString(py_init_params, "traceID")) != NULL) {
      csv_init_params->traceID_column = (gint) PyLong_AsLong(py_traceID);
    }

    if (((csvReader_init_params *) init_params)->has_header) DEBUG("csv data has header");

    DEBUG("delimiter %d(%c)\n", ((csvReader_init_params *) init_params)->delimiter,
          ((csvReader_init_params *) init_params)->delimiter);
  } else if (trace_type_str[0] == 'b') {
    trace_type = BIN_TRACE;
    if (!PyDict_Check(py_init_params)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "input init_params is not a valid python dictionary\n");
      return NULL;
    }

    init_params = g_new0(binary_init_params_t, 1);
    binary_init_params_t *bin_init_params = init_params;
    PyObject *py_label, *py_size, *py_op, *py_real_time, *py_fmt;

    if ((py_label = PyDict_GetItemString(py_init_params, "label")) != NULL) {
      bin_init_params->obj_id_pos = (gint) PyLong_AsLong(py_label);
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
      return NULL;
    }
    if (PyUnicode_READY(py_fmt) != 0) {
      PyErr_SetString(PyExc_RuntimeError,
                      "failed get fmt unicode ready\n");
      return NULL;
    }

    Py_UCS1 *py_ucs1 = PyUnicode_1BYTE_DATA(py_fmt);

    DEBUG("binary fmt %s\n", py_ucs1);
    strcpy(bin_init_params->fmt, (char *) py_ucs1);

  } else if (trace_type_str[0] == 'p') {
    trace_type = PLAIN_TXT_TRACE;
  } else if (trace_type_str[0] == 'v') {
    trace_type = VSCSI_TRACE;
    obj_id_type = OBJ_ID_NUM;
  } else {
    PyErr_SetString(PyExc_RuntimeError, "unknown trace type, supported trace types are c,p,b,v\n");
    return NULL;
  }

  reader_t *reader = setup_reader(file_loc, trace_type, obj_id_type, init_params);

  if (init_params != NULL) {
    g_free(init_params);
  }

  return PyCapsule_New((void *) reader, NULL, NULL);
  SUPPRESS_FUNCTION_NO_USE_WARNING(reader_pycapsule_destructor);
//    return PyCapsule_New((void *)reader, NULL, reader_pycapsule_destructor);
}


static PyObject *reader_read_one_req(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  request_t *req = new_req_struct(reader->base->obj_id_type);

  read_one_element(reader, req);
  if (req->valid) {
    if (req->obj_id_type == OBJ_ID_STR) {
      PyObject *ret = Py_BuildValue("s", (char *) (req->obj_id_ptr));
      destroy_req_struct(req);
      return ret;
    } else if (req->obj_id_type == OBJ_ID_NUM) {
      guint64 item = *((guint64 *) req->obj_id_ptr);
      destroy_req_struct(req);
      return Py_BuildValue("l", item);
    } else
      Py_RETURN_NONE;
  } else
    Py_RETURN_NONE;
}


static PyObject *reader_read_time_request(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  if (reader->base->trace_type == PLAIN_TXT_TRACE) {
    fprintf(stderr, "plain reader does not support get real time stamp\n");
    exit(1);
  }

  request_t *req = new_req_struct(reader->base->obj_id_type);

  read_one_element(reader, req);
  if (req->valid) {
    if (req->obj_id_type == OBJ_ID_STR) {
      PyObject *ret = Py_BuildValue("Ks", (unsigned long long) (req->real_time), (char *) (req->obj_id_ptr));
      destroy_req_struct(req);
      return ret;
    } else if (req->obj_id_type == OBJ_ID_NUM) {
      guint64 item = *((guint64 *) req->obj_id_ptr);
      destroy_req_struct(req);
      return Py_BuildValue("Kl", (unsigned long long) (req->real_time), item);
    } else
      Py_RETURN_NONE;
  } else
    Py_RETURN_NONE;
}

static PyObject *reader_read_complete_req(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  request_t *req = new_req_struct(reader->base->obj_id_type);

  read_one_element(reader, req);
  if (req->valid) {
    if (req->obj_id_type == OBJ_ID_STR) {
      PyObject *ret = Py_BuildValue("lsi", (long) req->real_time, (char *) (req->obj_id_ptr), (int) (req->size));
      destroy_req_struct(req);
      return ret;
    } else if (req->obj_id_type == OBJ_ID_NUM) {
      guint64 item = *((guint64 *) req->obj_id_ptr);
      PyObject *ret = Py_BuildValue("lli", (long) req->real_time, (long) item, (int) (req->size));
      destroy_req_struct(req);
      return ret;
    } else
      Py_RETURN_NONE;
  } else
    Py_RETURN_NONE;
}


static PyObject *reader_reset_reader(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  reset_reader(reader);

  Py_RETURN_NONE;
}

/* the following part is commented for now,
    bacause the reader here is not wrapped as a python object yet,
    and it also does not have enough functions to isolate as a new
    reader module to replace the old python one, but it needed,
    this module can be further wrapped as a python object
    Juncheng Yang 2016.5.27

*/


static PyObject *reader_reader_set_read_pos(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;
  float pos;

  // parse arguments
  if (!PyArg_ParseTuple(args, "Of", &po, &pos)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  reader_set_read_pos(reader, pos);
  Py_RETURN_NONE;
}


static PyObject *reader_get_num_of_req(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  long long num_of_req = get_num_of_req(reader);
  return Py_BuildValue("l", num_of_req);
}

static PyObject *reader_close_reader(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &po)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  int result = close_reader(reader);

  return Py_BuildValue("i", result);
}


static PyObject *reader_skip_N_requests(PyObject *self, PyObject *args) {
  reader_t *reader;
  PyObject *po;
  guint64 N;

  // parse arguments
  if (!PyArg_ParseTuple(args, "OI", &po, &N)) {
    return NULL;
  }
  if (!(reader = (reader_t *) PyCapsule_GetPointer(po, NULL))) {
    return NULL;
  }

  skip_N_elements(reader, N);

  Py_RETURN_NONE;
}


static PyMethodDef CacheReader_funcs[] = {
  {"setup_reader",      (PyCFunction) reader_setup_reader,
    METH_VARARGS | METH_KEYWORDS, "setup the c_reader in C extension for profiling"},
  {"get_num_of_req",    (PyCFunction) reader_get_num_of_req,
    METH_VARARGS, "return the number of requests in the cache file"},
  {"close_reader",      (PyCFunction) reader_close_reader,
    METH_VARARGS, "close c_reader"},
  {"read_one_req",      (PyCFunction) reader_read_one_req,
    METH_VARARGS, "read one element from reader"},
  {"reset_reader",      (PyCFunction) reader_reset_reader,
    METH_VARARGS, "reset reader to beginning position"},
  {"set_read_pos",      (PyCFunction) reader_reader_set_read_pos,
    METH_VARARGS, "set the next reading position of reader,\
        accept arguments like 0.5, 0.3 ..."},
  {"skip_N_requests",   (PyCFunction) reader_skip_N_requests,
    METH_VARARGS, "skip next N requests"},
  {"read_time_req",     (PyCFunction) reader_read_time_request,
    METH_VARARGS, "read one element with its real time from reader "
                  "in the form of tuple (real time, request)"},
  {"read_complete_req", (PyCFunction) reader_read_complete_req,
    METH_VARARGS, "read one element with its real time and size from reader"
                  " in the form of tuple (real time, request, size)"},

  {NULL}
};


static struct PyModuleDef CacheReader_definition = {
  PyModuleDef_HEAD_INIT,
  "CacheReader",
  "A Python module that creates and destroys a reader for fast profiling",
  -1,
  CacheReader_funcs
};


PyMODINIT_FUNC PyInit_CacheReader(void) {
  Py_Initialize();

  return PyModule_Create(&CacheReader_definition);
}

