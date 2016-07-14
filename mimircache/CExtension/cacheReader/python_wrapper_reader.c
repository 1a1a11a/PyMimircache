//
//  python_wrapper.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include <Python.h>
#include "reader.h"
#include "csvReader.h"
#include "const.h"


/* TODO: 
not urgent, necessary: add destructor method for reader py_capsule, so we don't need to call close reader
not urgent, not necessary: change this reader module into a pyhton object 
*/



static PyObject* reader_setup_reader(PyObject* self, PyObject* args, PyObject* keywds)
{   
    char* file_loc;
    char* file_type;
    PyObject *py_init_params;
    void *init_params = NULL;

    static char *kwlist[] = {"file_loc", "file_type", "init_params", NULL};
    
    // parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|O", kwlist, &file_loc, &file_type, &py_init_params)) {
        printf("parsing argument failed in setup reader\n");
        return NULL;
    }

    if (file_type[0] == 'c'){
        init_params = (void*) new_csvReader_init_params(-1, -1, -1, -1, FALSE, 0);
        /* if it is csv file, we need extra init parameters */
        PyObject *py_label, *py_size, *py_op, *py_real_time, *py_header, *py_delimiter;
        py_label = PyUnicode_FromString("label_column");
        py_size = PyUnicode_FromString("size_column");
        py_op = PyUnicode_FromString("op_column");
        py_real_time = PyUnicode_FromString("real_time_column");
        py_header = PyUnicode_FromString("header");
        py_delimiter = PyUnicode_FromString("delimiter");
        
        if (PyDict_Contains(py_init_params, py_label)){
            ((csvReader_init_params*)init_params)->label_column =
                                    (gint)PyLong_AsLong(PyDict_GetItemString(py_init_params, "label_column"));
            // release the PyObject?? 
        }
        
        
        if (PyDict_Contains(py_init_params, py_size)){
            ((csvReader_init_params*)init_params)->size_column =
                                    (gint)PyLong_AsLong(PyDict_GetItemString(py_init_params, "size_column"));
        }
        
        
        if (PyDict_Contains(py_init_params, py_op)){
            ((csvReader_init_params*)init_params)->op_column =
                                    (gint)PyLong_AsLong(PyDict_GetItemString(py_init_params, "op_column"));
        }
        
            
        if (PyDict_Contains(py_init_params, py_real_time)){
            ((csvReader_init_params*)init_params)->real_time_column =
                                    (gint)PyLong_AsLong(PyDict_GetItemString(py_init_params, "real_time_column"));
        }
        
        if (PyDict_Contains(py_init_params, py_header)){
            ((csvReader_init_params*)init_params)->has_header =
            PyObject_IsTrue(PyDict_GetItemString(py_init_params, "header"));
        }
        
        if (PyDict_Contains(py_init_params, py_delimiter)){
            ((csvReader_init_params*)init_params)->delimiter =
            *((unsigned char*)PyUnicode_AsUTF8(PyDict_GetItemString(py_init_params, "delimiter")));
        }

        
        Py_DECREF(py_label);
        Py_DECREF(py_size);
        Py_DECREF(py_op);
        Py_DECREF(py_real_time);
        Py_DECREF(py_header);
        Py_DECREF(py_delimiter);
        
        if (((csvReader_init_params*)init_params)->has_header)
            DEBUG(printf("has header\n"));
        
        if (((csvReader_init_params*)init_params)->delimiter)
            DEBUG(printf("delimiter %c\n", ((csvReader_init_params*)init_params)->delimiter));
        
        
    }
    
    READER* reader = setup_reader(file_loc, *file_type, init_params);

    if (init_params != NULL){
        g_free(init_params);
    }
    return PyCapsule_New((void *)reader, NULL, NULL);
}


static PyObject* reader_read_one_element(PyObject* self, PyObject* args)
{
    READER* reader;
    PyObject* po; 
    cache_line* c = new_cacheline();
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    } 

    if (reader->type == 'p' || reader->type == 'c')
        c->type = 'c';
    else if (reader->type == 'v')
        c->type = 'l';
    else{
        printf("reader type not recognized: %c\n", reader->type);
        exit(1);
    }

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

static PyObject* reader_reset_reader(PyObject* self, PyObject* args)
{
    READER* reader;
    PyObject* po; 
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
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



static PyObject* reader_skip_N_elements(PyObject* self)
{
    return Py_BuildValue("s", "Hello, Python extensions!!");
}





*/


static PyObject* reader_reader_set_read_pos(PyObject* self, PyObject* args)
{
    READER* reader;
    PyObject* po;
    float pos;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "Of", &po, &pos)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }
    
    reader_set_read_pos(reader, pos);
    Py_RETURN_NONE;
}


static PyObject* reader_get_num_of_cache_lines(PyObject* self, PyObject* args)
{   
    READER* reader;
    PyObject* po; 
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    } 

    long long num_of_lines = get_num_of_cache_lines(reader);
    return Py_BuildValue("l", num_of_lines);
}

static PyObject* reader_close_reader(PyObject* self, PyObject* args)
{
    READER* reader;
    PyObject* po;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "O", &po)) {
        return NULL;
    }
    if (!(reader = (READER*) PyCapsule_GetPointer(po, NULL))) {
        return NULL;
    }    

    int result = close_reader(reader);    

    return Py_BuildValue("i", result);
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

