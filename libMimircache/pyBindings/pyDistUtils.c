//
// Created by Juncheng Yang on 11/24/19.
//


#include "pyHeaders.h"




static PyObject* py_get_last_access_dist(PyObject* self, PyObject* args, PyObject* keywds) {
  PyObject* po;
  reader_t* reader;
  static char *kwlist[] = {"reader", NULL};

  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &po)) {
    PyErr_SetString(PyExc_RuntimeError, "parsing argument failed");
    return NULL;
  }
  if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
    PyErr_SetString(PyExc_RuntimeError, "error retrieving reader capsule");
    return NULL;
  }

  // get last access dist list
  gint64 *dist_array = get_last_access_dist(reader);

  // create numpy array
  npy_intp dims[1] = { get_num_of_req(reader) };
  PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_INT);
  int* array = (int*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0);
  for (guint64 i=0; i<get_num_of_req(reader); i++){
    array[i] = dist_array[i];
  }
  return ret_array;
}


static PyObject* py_get_next_access_dist_seq(PyObject* self, PyObject* args, PyObject* keywds) {
  PyObject* po;
  reader_t* reader;
  static char *kwlist[] = {"reader", NULL};
  // parse arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &po)) {
    PyErr_SetString(PyExc_RuntimeError, "parsing argument failed");
    return NULL;
  }
  if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
    PyErr_SetString(PyExc_RuntimeError, "error retrieving reader capsule");
    return NULL;
  }

  // get reversed last access dist list
  gint64 *dist_array = get_next_access_dist(reader);

  // create numpy array
  npy_intp dims[1] = { get_num_of_req(reader) };
  PyObject* ret_array = PyArray_SimpleNew(1, dims, NPY_INT);
  int* array = (int*) PyArray_GETPTR1((PyArrayObject *)ret_array, 0);
  for (guint64 i=0; i<get_num_of_req(reader); i++){
    array[i] = dist_array[i];
  }
  return ret_array;
}


static PyMethodDef DistUtils_funcs[] = {
  {"get_last_access_dist", (PyCFunction) py_get_last_access_dist,
   METH_VARARGS | METH_KEYWORDS, "get the distance to the last access, -1 if haven't seen before"},
  {"get_next_access_dist", (PyCFunction) py_get_next_access_dist_seq,
   METH_VARARGS | METH_KEYWORDS, "get the distance to the next access, -1 if it won't be accessed any more"},
//  {"get_breakpoints",      (PyCFunction) py_get_break_points,
//   METH_VARARGS | METH_KEYWORDS, "generate virtual/real break points"},
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef DistUtils_definition = {
  PyModuleDef_HEAD_INIT,
  "DistUtils",
  "A Python module that performs distance related computation",
  -1,
  DistUtils_funcs
};



PyMODINIT_FUNC PyInit_DistUtils(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&DistUtils_definition);
}

