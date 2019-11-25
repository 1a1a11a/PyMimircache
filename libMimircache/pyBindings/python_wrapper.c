//
//  python_wrapper.c
//  pyMimircache
//
//  Created by Juncheng on 5/24/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//



#include <LRU_K.h>
#include "python_wrapper.h"
#include "../libMimircache/libMimircache/include/mimircache/plugin.h"
#include "../libMimircache/libMimircache/cache/include/cacheHeaders.h"

static void reader_pycapsule_destructor(PyObject *pycap_reader){
  reader_t *reader = (reader_t*) PyCapsule_GetPointer(pycap_reader, NULL);
  close_reader(reader);
}

cache_t *build_cache(reader_t *reader, long cache_size, char *algorithm, PyObject *cache_params, long begin) {
  cache_t *cache;
  void *cache_init_params = NULL;
//    int block_unit_size = 0;
//    if (cache_params != NULL && cache_params!=Py_None && PyDict_Check(cache_params) &&
//        PyDict_Contains(cache_params, PyUnicode_FromString("block_unit_size"))){
//        block_unit_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_unit_size"));
//        INFO("considering cache size in profiling\n");
//    }

  if (strcmp(algorithm, "Optimal") == 0) {
    Optimal_init_params_t *init_params = g_new(Optimal_init_params_t, 1);
    init_params->ts = begin;
    init_params->reader = reader;
    init_params->next_access = NULL;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "LRU_2") == 0) {
    LRU_K_init_params_t *init_params = g_new(LRU_K_init_params_t, 1);
    init_params->K = 2;
    init_params->maxK = 2;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "LRU_K") == 0) {
    int K = (int) PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
    LRU_K_init_params_t *init_params = g_new(LRU_K_init_params_t, 1);
    init_params->K = K;
    init_params->maxK = K;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "ARC") == 0) {
    ARC_init_params_t *init_params = g_new(ARC_init_params_t, 1);
    if (cache_params != Py_None && PyDict_Contains(cache_params, PyUnicode_FromString("ghost_list_factor")))
      init_params->ghost_list_factor = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "ghost_list_factor"));
    else
      init_params->ghost_list_factor = 10;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "SLRU") == 0) {
    SLRU_init_params_t *init_params = g_new(struct SLRU_init_params, 1);
    if (cache_params != Py_None && PyDict_Contains(cache_params, PyUnicode_FromString("N")))
      init_params->N_segments = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "N"));
    else
      init_params->N_segments = 2;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "PG") == 0) {
    PG_init_params_t *init_params = g_new(PG_init_params_t, 1);
    init_params->lookahead = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "lookahead"));
    init_params->block_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_size"));
    init_params->max_meta_data = (double) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "max_metadata_size"));
    init_params->prefetch_threshold = (double) PyFloat_AsDouble(
      PyDict_GetItemString(cache_params, "prefetch_threshold"));

    PyObject *temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8",
                                                     "strict"); // Owned reference
    if (temp_bytes != NULL) {
      init_params->cache_type = g_strdup(PyBytes_AS_STRING(temp_bytes)); // Borrowed pointer
      Py_DECREF(temp_bytes);
    }
    DEBUG("PG lookahead %d, max_meta_data %lf, prefetch_threshold %lf, cache type %s\n",
          init_params->lookahead, init_params->max_meta_data,
          init_params->prefetch_threshold, init_params->cache_type);
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "AMP") == 0) {
    gint threshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "pthreshold"));
    gint K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
    DEBUG("AMP K %d, threshold %d\n", K, threshold);
    struct AMP_init_params *init_params = g_new(struct AMP_init_params, 1);
    init_params->APT = 4;
    init_params->read_size = 1;
    init_params->p_threshold = threshold;
    init_params->K = K;
    cache_init_params = init_params;
  } else if (strcmp(algorithm, "Mithril") == 0) {
    gint max_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "max_support"));
    gint min_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "min_support"));
    gint lookahead_range = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "lookahead_range"));
    gint prefetch_list_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_list_size"));
    gdouble max_metadata_size = (gdouble) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "max_metadata_size"));
    gint block_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_size"));

    gint cycle_time = 2;
    if (PyDict_Contains(cache_params, PyUnicode_FromString("cycle_time")))
      cycle_time = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "cycle_time"));

    gint sequential_type = 0;
    if (PyDict_Contains(cache_params, PyUnicode_FromString("sequential_type")))
      sequential_type = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_type"));

    gint confidence = 1;
    if (PyDict_Contains(cache_params, PyUnicode_FromString("confidence")))
      confidence = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "confidence"));

    gint mining_threshold = MINING_THRESHOLD;
    if (PyDict_Contains(cache_params, PyUnicode_FromString("mining_threshold")))
      mining_threshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "mining_threshold"));

    gint sequential_K = -1;
    if (sequential_type != 0)
      sequential_K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_K"));


    gchar *cache_type = "LRU";
    if (PyDict_Contains(cache_params, PyUnicode_FromString("cache_type"))) {
      PyObject *temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8",
                                                       "strict"); // Owned reference
      if (temp_bytes != NULL) {
        cache_type = g_strdup(PyBytes_AS_STRING(temp_bytes)); // Borrowed pointer
        Py_DECREF(temp_bytes);
      }
    }
    if (strcmp(cache_type, "unknown") == 0) {
      PyErr_SetString(PyExc_RuntimeError, "please provide cache_type\n");
      return NULL;
    }

    rec_trigger_e rec_trigger = miss;
    if (PyDict_Contains(cache_params, PyUnicode_FromString("rec_trigger"))) {
      PyObject *temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "rec_trigger"), "utf-8",
                                                       "strict"); // Owned reference
      if (temp_bytes != NULL) {
        // Default is record at miss if not set
        if (strcmp(PyBytes_AS_STRING(temp_bytes), "miss") == 0)
          rec_trigger = miss;
        else if (strcmp(PyBytes_AS_STRING(temp_bytes), "evict") == 0)
          rec_trigger = evict;
        else if (strcmp(PyBytes_AS_STRING(temp_bytes), "miss_evict") == 0)
          rec_trigger = miss_evict;
        else if (strcmp(PyBytes_AS_STRING(temp_bytes), "each_req") == 0)
          rec_trigger = each_req;
        else ERROR("unknown recording loc %s\n", PyBytes_AS_STRING(temp_bytes));
        Py_DECREF(temp_bytes);
      }
    }

    gint AMP_pthreshold = -1;
    if (strcmp(cache_type, "AMP") == 0)
      AMP_pthreshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "AMP_pthreshold"));

    DEBUG("cache type %s, max support=%d, min support %d, confidence %d, sequential %d\n",
          cache_type, max_support, min_support, confidence, sequential_K);

    Mithril_init_params_t *init_params = g_new(Mithril_init_params_t, 1);
    init_params->max_support = max_support;
    init_params->min_support = min_support;
    init_params->cache_type = cache_type;
    init_params->confidence = confidence;
    init_params->rec_trigger = rec_trigger;
    init_params->lookahead_range = lookahead_range;
    init_params->pf_list_size = prefetch_list_size;
    init_params->block_size = block_size;
    init_params->max_metadata_size = max_metadata_size;
    init_params->sequential_type = sequential_type;
    init_params->sequential_K = sequential_K;
    init_params->AMP_pthreshold = AMP_pthreshold;
    init_params->cycle_time = cycle_time;
    init_params->mining_threshold = mining_threshold;
    cache_init_params = init_params;
  } else {
    PyErr_Format(PyExc_RuntimeError,
                 "does not support given cache replacement algorithm: %s\n", algorithm);
    return NULL;
  }
  cache = create_cache(algorithm, cache_size, reader->base->obj_id_type, cache_init_params);

  return cache;
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

