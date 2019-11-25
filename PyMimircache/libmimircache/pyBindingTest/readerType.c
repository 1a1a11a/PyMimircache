#include <Python.h> 
// #include <reader.h> 



typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    // reader_t reader_struct; 
    
} mimircache_ReaderObject;

static PyTypeObject mimircache_ReaderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "mimircache.reader",             /* tp_name */
    sizeof(mimircache_ReaderObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "mimircache Reader",           /* tp_doc */
};

static PyModuleDef mimircachemodule = {
    PyModuleDef_HEAD_INIT,
    "reader",
    "Example module that creates an extension type.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_mimircache(void)
{
    PyObject* m;

    mimircache_ReaderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&mimircache_ReaderType) < 0)
        return NULL;

    m = PyModule_Create(&mimircachemodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&mimircache_ReaderType);
    PyModule_AddObject(m, "reader", (PyObject *)&mimircache_ReaderType);
    return m;
}