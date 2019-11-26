//
//  pyHeaders.h
//  PyMimircache
//
//  Created by Juncheng on 5/24/16.
//  Refactored by Juncheng on 11/26/19.
//  Copyright © 2016-2019 Juncheng. All rights reserved.
//

#ifndef pyHeaders_h
#define pyHeaders_h




// python3-config --include
// np.get_include()

#define NPY_NO_DEPRECATED_API 11


#if defined(__linux__) || defined(__APPLE__)
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
//#elif __APPLE__
//#include "/usr/local/Cellar/python/3.7.5/Frameworks/Python.framework/Versions/3.7/include/python3.7m/Python.h"
//#include "/Users/jason/Library/Python/3.7/lib/python/site-packages/numpy/core/include/numpy/arrayobject.h"
//#include "/Users/jason/Library/Python/3.7/lib/python/site-packages/numpy/core/include/numpy/npy_math.h"
#elif _WIN32
#warning "libMimircache does not support windows sorry"
#endif




#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <math.h>

#include "../libMimircache/libMimircache/include/mimircache.h"
#include "../libMimircache/libMimircache/cache/include/cacheHeaders.h"


static inline cache_t *py_create_cache(char *algorithm, long cache_size, obj_id_t obj_id_type, PyObject *cache_params, long begin){
    cache_t *cache;
    void *cache_init_params = NULL;

    if (strcmp(algorithm, "Optimal") == 0) {
      Optimal_init_params_t *init_params = g_new(Optimal_init_params_t, 1);
      init_params->ts = begin;
//      init_params->reader = reader;
      PyObject* po = PyDict_GetItemString(cache_params, "reader");
      if (po == NULL){
        ERROR("Optimal algorithm requires reader in init_params\n");
        PyErr_SetString(PyExc_RuntimeError, "Optimal algorithm requires reader in init_params");
        exit(1);
      }
      init_params->reader = (reader_t *) PyCapsule_GetPointer(po, NULL);
      if (init_params->reader == NULL){
        ERROR("error retrieval reader from capsule in %s\n", __func__);
        PyErr_SetString(PyExc_RuntimeError, "error retrieval reader from capsule");
        return NULL;
      }
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
    cache = create_cache(algorithm, cache_size, obj_id_type, cache_init_params);

    return cache;
}


//static PyObject *py_setup_reader(PyObject *self, PyObject *args, PyObject *keywds);






#endif /* pyHeaders_h */



