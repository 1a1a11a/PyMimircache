//
//  python_wrapper.c
//  python_wrapper
//
//  Created by Juncheng on 5/24/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//



#include "python_wrapper.h" 




struct_cache* build_cache(reader_t* reader,
                          long cache_size,
                          char* algorithm,
                          PyObject* cache_params,
                          long begin){

    struct_cache *cache;
    char data_type = reader->base->data_type;
    int block_unit_size = 0;
    if (cache_params != NULL && cache_params!=Py_None && PyDict_Check(cache_params) &&
        PyDict_Contains(cache_params, PyUnicode_FromString("block_unit_size"))){
        block_unit_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_unit_size"));
        INFO("considering cache size in profiling\n");
    }
    

    
    
    if (strcmp(algorithm, "FIFO") == 0){
        cache = fifo_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "LRU") == 0){
        // WARNING("we suggest using LRUProfiler for profiling, it's faster\n");
        cache = LRU_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "LFU") == 0){
        cache = LFU_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "LFUFast") == 0){
        cache = LFU_fast_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "MRU") == 0){
        cache = MRU_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "Random") == 0){
        cache = Random_init(cache_size, data_type, block_unit_size, NULL);
    }
    else if (strcmp(algorithm, "Optimal") == 0){
        struct optimal_init_params *init_params = g_new(struct optimal_init_params, 1);
        init_params->ts = begin;
        init_params->reader = reader;
        init_params->next_access = NULL;
        cache = optimal_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
    else if (strcmp(algorithm, "LRU_2") == 0){
        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
        init_params->K = 2;
        init_params->maxK = 2;
        cache = LRU_K_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
    else if (strcmp(algorithm, "ARC") == 0){
        ARC_init_params_t *init_params = g_new(ARC_init_params_t, 1);
        if (cache_params!=Py_None && PyDict_Contains(cache_params, PyUnicode_FromString("ghost_list_factor")))
            init_params->ghost_list_factor = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "ghost_list_factor"));
        else
            init_params->ghost_list_factor = 10;
        cache = ARC_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
//    else if (strcmp(algorithm, "LRFU") == 0){
//        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
//        init_params->K = 2;
//        init_params->maxK = 2;
//        cache = ARC_init(cache_size, data_type, (void*)init_params);
//    }
    else if (strcmp(algorithm, "SLRU") == 0){
        SLRU_init_params_t *init_params = g_new(struct SLRU_init_params, 1);
        if (cache_params!=Py_None && PyDict_Contains(cache_params, PyUnicode_FromString("N")))
            init_params->N_segments = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "N"));
        else
            init_params->N_segments = 2;
        cache = SLRU_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }

#ifdef ML
    else if (strcmp(algorithm, "SLRUML") == 0){
        SLRUML_init_params_t *init_params = g_new0(SLRUML_init_params_t, 1);
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "hint_loc"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            strcpy(init_params->hint_loc, PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
            DEBUG_MSG("SLRUML hint_loc %s\n", init_params->hint_loc);
        }
        else{
            strcpy(init_params->hint_loc, "hint");
        }

        init_params->N_segments = 5;
        cache = SLRUML_init(cache_size, data_type, block_unit_size, init_params);
    }

    else if (strcmp(algorithm, "ScoreML") == 0){
        Score_init_params_t *init_params = g_new0(Score_init_params_t, 1);
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "hint_loc"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            strcpy(init_params->hint_loc, PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
            DEBUG_MSG("ScoreML hint_loc %s\n", init_params->hint_loc);
        }
        else{
            strcpy(init_params->hint_loc, "hint");
        }
        
        cache = Score_init(cache_size, data_type, block_unit_size, init_params);
    }
#endif 

    else if (strcmp(algorithm, "LRU_K") == 0){
        int K = (int)PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
        init_params->K = K;
        init_params->maxK = K;
        cache = LRU_K_init(cache_size, data_type, block_unit_size, (void*)init_params);
        DEBUG_MSG("cache->K = %d, maxK = %d\n",
                  ((struct LRU_K_params*)(cache->cache_params))->K,
                  ((struct LRU_K_params*)(cache->cache_params))->maxK);
        
    }
    else if (strcmp(algorithm, "PG") == 0){
        PG_init_params_t *init_params = g_new(PG_init_params_t, 1);
        init_params->lookahead          = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "lookahead"));
        init_params->block_size         = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_size"));
        init_params->max_meta_data      = (double) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "max_metadata_size"));
        init_params->prefetch_threshold = (double) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "prefetch_threshold"));
        
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            init_params->cache_type = g_strdup( PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        DEBUG_MSG("PG lookahead %d, max_meta_data %lf, prefetch_threshold %lf, cache type %s\n",
                  init_params->lookahead, init_params->max_meta_data,
                  init_params->prefetch_threshold, init_params->cache_type);
        cache = PG_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
    else if (strcmp(algorithm, "AMP") == 0){
        gint threshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "pthreshold"));
        gint K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
        DEBUG_MSG("AMP K %d, threshold %d\n", K, threshold);
        struct AMP_init_params *init_params = g_new(struct AMP_init_params, 1);
        init_params->APT            = 4;
        init_params->read_size      = 1;
        init_params->p_threshold    = threshold;
        init_params->K              = K;
        cache = AMP_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
    else if (strcmp(algorithm, "mimir") == 0){
        gint max_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "max_support"));
        gint min_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "min_support"));
        gint confidence = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "confidence"));
        gint item_set_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "item_set_size"));
        gint prefetch_list_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_list_size"));
        gint sequential_type = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_type"));
        gdouble max_metadata_size = (gdouble) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "max_metadata_size"));
        gint block_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "block_size"));
        gint cycle_time = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "cycle_time"));
        gint mining_threshold = 5120;
        if (PyDict_Contains(cache_params, PyUnicode_FromString("mining_threshold")))
            mining_threshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "mining_threshold"));

        gint AMP_pthreshold = -1;
        recording_loc_e recording_loc = miss;
        
        gint sequential_K = -1;
        if (sequential_type != 0)
            sequential_K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_K"));
        gchar* cache_type = "unknown";
        
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            cache_type = g_strdup( PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        
        temp_bytes = NULL;
        if (PyDict_Contains(cache_params, PyUnicode_FromString("recording_loc")))
            temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "recording_loc"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            // Default is record at miss if not set 
            if (strcmp(PyBytes_AS_STRING(temp_bytes), "miss") == 0)
                recording_loc = miss;
            else if (strcmp(PyBytes_AS_STRING(temp_bytes), "evict") == 0)
                recording_loc = evict;
            else if (strcmp(PyBytes_AS_STRING(temp_bytes), "miss-evict") == 0)
                recording_loc = miss_evict;
            else if (strcmp(PyBytes_AS_STRING(temp_bytes), "each-req") == 0)
                recording_loc = each_req; 
            else
                ERROR("unknown recording loc %s\n", PyBytes_AS_STRING(temp_bytes));
            Py_DECREF(temp_bytes);
        }

        
        if (strcmp(cache_type, "unknown") == 0){
            PyErr_SetString(PyExc_RuntimeError, "please provide cache_type\n");
            return NULL;
        }
        if (strcmp(cache_type, "AMP") == 0)
            AMP_pthreshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "AMP_pthreshold"));
        
        DEBUG_MSG("cache type %s, max support=%d, min support %d, confidence %d, sequential %d\n",
                     cache_type, max_support, min_support, confidence, sequential_K);
        
        struct MIMIR_init_params *init_params = g_new(struct MIMIR_init_params, 1);
        init_params->max_support            = max_support;
        init_params->min_support            = min_support;
        init_params->cache_type             = cache_type;
        init_params->confidence             = confidence;
        init_params->recording_loc          = recording_loc;
        init_params->item_set_size          = item_set_size;
        init_params->prefetch_list_size     = prefetch_list_size;
        init_params->block_size             = block_size;
        init_params->max_metadata_size      = max_metadata_size;
        init_params->sequential_type        = sequential_type;
        init_params->sequential_K           = sequential_K;
        init_params->AMP_pthreshold         = AMP_pthreshold;
        init_params->cycle_time             = cycle_time;
        init_params->mining_threshold       = mining_threshold;
        
        cache = MIMIR_init(cache_size, data_type, block_unit_size, (void*)init_params);
    }
    else {
        PyErr_Format(PyExc_RuntimeError,
                        "does not support given cache replacement algorithm: %s\n", algorithm);
        return NULL;
    }
    
    
    return cache;
}
