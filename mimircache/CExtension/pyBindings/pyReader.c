//
//  python_wrapper.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>
#include "const.h"
#include "reader.h"
#include "csvReader.h"
#include "binaryReader.h"


/* TODO: 
not urgent, necessary: add destructor method for reader py_capsule, so we don't need to call close reader
not urgent, not necessary: change this reader module into a pyhton object 
*/

static void reader_pycapsule_destructor(PyObject *pycap_reader){
    reader_t *reader = (reader_t*) PyCapsule_GetPointer(pycap_reader, NULL);
    close_reader(reader);
}

static PyObject* reader_setup_reader(PyObject* self, PyObject* args, PyObject* keywds)
{   
    char* file_loc;
    char* file_type;
    char* data_type = "c";
    int block_unit_size = 0;
    int disk_sector_size = 0;
    PyObject *py_init_params;
    void *init_params = NULL;

    static char *kwlist[] = {"file_loc", "file_type", "data_type",
                "block_unit_size", "disk_sector_size", "init_params", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|siiO", kwlist, &file_loc,
                                     &file_type, &data_type, &block_unit_size,
                                     &disk_sector_size, &py_init_params)){
        PyErr_SetString(PyExc_RuntimeError,
                        "parsing argument failed in setup reader\n");
        return NULL;
    }
    
    if (file_type[0] == 'v')
        data_type = "l"; 

    if (file_type[0] == 'c'){
        if (!PyDict_Check(py_init_params)){
            PyErr_SetString(PyExc_RuntimeError,
                            "input init_params is not a valid python dictionary\n");
            return NULL;
        }
        init_params = (void*) new_csvReader_init_params(-1, -1, -1, -1, FALSE, ',', -1);
        /* if it is csv file, we need extra init parameters */
        PyObject *py_label, *py_size, *py_op, *py_real_time,
                    *py_header, *py_delimiter, *py_traceID;
        csvReader_init_params* csv_init_params = init_params;
        
        if ( (py_label = PyDict_GetItemString(py_init_params, "label_column")) != NULL){
            csv_init_params->label_column = (gint)PyLong_AsLong(py_label);
        }
        
        if ( (py_size = PyDict_GetItemString(py_init_params, "size_column")) != NULL) {
            csv_init_params->size_column = (gint)PyLong_AsLong(py_size);
        }
        
        if ( (py_op = PyDict_GetItemString(py_init_params, "op_column")) != NULL) {
            csv_init_params->op_column = (gint)PyLong_AsLong(py_op);
        }
        
        if ( (py_real_time = PyDict_GetItemString(py_init_params, "real_time_column")) != NULL) {
            csv_init_params->real_time_column = (gint)PyLong_AsLong(py_real_time);
        }
        
        if ( (py_header = PyDict_GetItemString(py_init_params, "header")) != NULL) {
            csv_init_params->has_header = PyObject_IsTrue(py_header);
        }
        
        if ( (py_delimiter = PyDict_GetItemString(py_init_params, "delimiter")) != NULL) {
            csv_init_params->delimiter = *((unsigned char*)PyUnicode_AsUTF8(py_delimiter));
        }
        
        if ( (py_traceID = PyDict_GetItemString(py_init_params, "traceID_column")) != NULL) {
            csv_init_params->traceID_column = (gint)PyLong_AsLong(py_traceID);
        }
        

        if (((csvReader_init_params*)init_params)->has_header)
            DEBUG_MSG("csv data has header");
        
        DEBUG_MSG("delimiter %d(%c)\n", ((csvReader_init_params*)init_params)->delimiter,
                  ((csvReader_init_params*)init_params)->delimiter);
    }
    
    
    else if (file_type[0] == 'b'){
        if (!PyDict_Check(py_init_params)){
            PyErr_SetString(PyExc_RuntimeError,
                            "input init_params is not a valid python dictionary\n");
            return NULL;
        }
        
        init_params = g_new0(binary_init_params_t, 1);
        binary_init_params_t *bin_init_params = init_params;
        PyObject *py_label, *py_size, *py_op, *py_real_time, *py_fmt;
        
        if ( (py_label = PyDict_GetItemString(py_init_params, "label")) != NULL){
            bin_init_params->label_pos = (gint)PyLong_AsLong(py_label);
        }
        
        if ( (py_size = PyDict_GetItemString(py_init_params, "size")) != NULL) {
            bin_init_params->size_pos = (gint)PyLong_AsLong(py_size);
        }
        
        if ( (py_op = PyDict_GetItemString(py_init_params, "op")) != NULL) {
            bin_init_params->op_pos = (gint)PyLong_AsLong(py_op);
        }
        
        if ( (py_real_time = PyDict_GetItemString(py_init_params, "real_time")) != NULL) {
            bin_init_params->real_time_pos = (gint)PyLong_AsLong(py_real_time);
        }

        
        py_fmt = PyDict_GetItemString(py_init_params, "fmt");
        if (!PyUnicode_Check(py_fmt)){
            PyErr_SetString(PyExc_RuntimeError,
                            "passed format string is not unicode \n");
            return NULL;
        }
        if (PyUnicode_READY(py_fmt) != 0){
            PyErr_SetString(PyExc_RuntimeError,
                            "failed get fmt unicode ready\n");
            return NULL;
        }
            
        Py_UCS1* py_ucs1 = PyUnicode_1BYTE_DATA(py_fmt);
        
        DEBUG_MSG("binary fmt %s\n", py_ucs1);
        strcpy(bin_init_params->fmt, (char*) py_ucs1);
        
    }
    
    
    reader_t* reader = setup_reader(file_loc, *file_type, *data_type,
                                    block_unit_size, disk_sector_size, init_params);

    if (init_params != NULL){
        g_free(init_params);
    }
    
    return PyCapsule_New((void *)reader, NULL, NULL);
    SUPPRESS_FUNCTION_NO_USE_WARNING(reader_pycapsule_destructor);
//    return PyCapsule_New((void *)reader, NULL, reader_pycapsule_destructor); 
}


static PyObject* reader_read_one_element(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po; 
    cache_line* c = new_cacheline();
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    } 

    c->type = reader->base->data_type;

    read_one_element(reader, c);
    if (c->valid){
        if (c->type == 'c'){
            PyObject* ret = Py_BuildValue("s", (char*)(c->item_p));
            destroy_cacheline(c);
            return ret;
        }
        else if (c->type == 'l'){
            guint64 item = *((guint64*)c->item_p);
            destroy_cacheline(c);
            return Py_BuildValue("l", item);
        }
        else
            Py_RETURN_NONE;
        }
    else
        Py_RETURN_NONE;
}


static PyObject* reader_read_time_request(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po;
    cache_line* c = new_cacheline();
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (reader->base->type == 'p'){
        fprintf(stderr, "plain reader does not support get real time stamp\n");
        exit(1);
    }
    else{
        c->type = reader->base->data_type;
    }
    
    read_one_element(reader, c);
    if (c->valid){
        if (c->type == 'c'){
            PyObject* ret = Py_BuildValue("Ks", (unsigned long long)(c->real_time), (char*)(c->item_p));
            destroy_cacheline(c);
            return ret;
        }
        else if (c->type == 'l'){
            guint64 item = *((guint64*)c->item_p);
            destroy_cacheline(c);
            return Py_BuildValue("Kl", (unsigned long long)(c->real_time), item);
        }
        else
            Py_RETURN_NONE;
    }
    else
        Py_RETURN_NONE;
}

static PyObject* reader_read_one_request_full_info(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po;
    cache_line* c = new_cacheline();
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    if (reader->base->type == 'p'){
        fprintf(stderr, "plain reader does not support get full info\n");
        exit(1);
    }
    else{
        c->type = reader->base->data_type;
    }
    
    read_one_element(reader, c);
    if (c->valid){
        if (c->type == 'c'){
            PyObject* ret = Py_BuildValue("lsi", (long)c->real_time, (char*)(c->item_p), (int)(c->size));
            destroy_cacheline(c);
            return ret;
        }
        else if (c->type == 'l'){
            guint64 item = *((guint64*)c->item_p);
            PyObject* ret = Py_BuildValue("lli", (long)c->real_time, (long)item, (int)(c->size));
            destroy_cacheline(c);
            return ret;
        }
        else
            Py_RETURN_NONE;
    }
    else
        Py_RETURN_NONE;
}





static PyObject* reader_reset_reader(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po; 
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
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


static PyObject* reader_reader_set_read_pos(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po;
    float pos;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "Of", &po, &pos)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    reader_set_read_pos(reader, pos);
    Py_RETURN_NONE;
}


static PyObject* reader_get_num_of_cache_lines(PyObject* self, PyObject* args)
{   
    reader_t* reader;
    PyObject* po; 
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    } 

    long long num_of_lines = get_num_of_cache_lines(reader);
    return Py_BuildValue("l", num_of_lines);
}

static PyObject* reader_close_reader(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }    

    int result = close_reader(reader);    

    return Py_BuildValue("i", result);
}


static PyObject* reader_skip_N_requests(PyObject* self, PyObject* args)
{
    reader_t* reader;
    PyObject* po;
    guint64 N;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "OI", &po, &N)) {
        return NULL;
    }
    if (!(reader = (reader_t*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    skip_N_elements(reader, N);
    
    Py_RETURN_NONE;
}



static PyMethodDef c_cacheReader_funcs[] = {
    {"setup_reader", (PyCFunction)reader_setup_reader,
        METH_VARARGS | METH_KEYWORDS, "setup the c_reader in C extension for profiling"},
    {"get_num_of_lines", (PyCFunction)reader_get_num_of_cache_lines,
        METH_VARARGS, "return the number of requests in the cache file"},
    {"close_reader", (PyCFunction)reader_close_reader,
        METH_VARARGS, "close c_reader"},
    {"read_one_element", (PyCFunction)reader_read_one_element,
        METH_VARARGS, "read one element from reader"},
    {"reset_reader", (PyCFunction)reader_reset_reader,
        METH_VARARGS, "reset reader to beginning position"},
    {"set_read_pos", (PyCFunction)reader_reader_set_read_pos,
        METH_VARARGS, "set the next reading position of reader,\
        accept arguments like 0.5, 0.3 ..."},
    {"skip_N_requests", (PyCFunction)reader_skip_N_requests,
        METH_VARARGS, "skip next N requests"},
    {"read_time_request", (PyCFunction)reader_read_time_request,
        METH_VARARGS, "read one element with its real time from reader "
        "in the form of tuple (real time, request)"},
    {"read_one_request_full_info", (PyCFunction)reader_read_one_request_full_info,
        METH_VARARGS, "read one element with its real time and size from reader"
        " in the form of tuple (real time, request, size)"},
    
    {NULL}
};


static struct PyModuleDef c_cacheReader_definition = { 
    PyModuleDef_HEAD_INIT,
    "c_cacheReader",
    "A Python module that creates and destroys a reader for fast profiling",
    -1, 
    c_cacheReader_funcs
};



PyMODINIT_FUNC PyInit_c_cacheReader(void)
{
    Py_Initialize();

    return PyModule_Create(&c_cacheReader_definition);
}

