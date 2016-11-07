


#include "python_wrapper.h" 




struct_cache* build_cache(READER* reader, long cache_size, char* algorithm, PyObject* cache_params, long begin){

    struct_cache *cache;
    char data_type = reader->data_type;
    
    if (strcmp(algorithm, "FIFO") == 0){
        cache = fifo_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "LRU") == 0){
        printf("we suggest using LRUProfiler for profiling, it's faster\n");
        cache = LRU_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "LFU") == 0){
        cache = LFU_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "MRU") == 0){
        cache = MRU_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "Random") == 0){
        cache = Random_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "LRU_LFU") == 0){
        double LRU_percentage = PyFloat_AS_DOUBLE(PyDict_GetItemString(cache_params, "LRU_percentage"));
#ifdef DEBUG
    printf("LRU_percentage=%lf\n", LRU_percentage);
#endif
        struct LRU_LFU_init_params *init_params = g_new(struct LRU_LFU_init_params, 1);
        init_params->LRU_percentage = LRU_percentage;
        cache = LRU_LFU_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "LRU_dataAware") == 0){
        cache = LRU_dataAware_init(cache_size, data_type, NULL);
    }
    else if (strcmp(algorithm, "Optimal") == 0){
        struct optimal_init_params *init_params = g_new(struct optimal_init_params, 1);
        init_params->ts = begin;
        init_params->reader = reader;
        init_params->next_access = NULL;
        cache = optimal_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "LRU_2") == 0){
        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
        init_params->K = 2;
        init_params->maxK = 2;
        cache = LRU_K_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "LRU_K") == 0){
//        printf("check dict\n");
//        printf("check = %d\n", PyDict_Check(cache_params));
//        printf("prepare LRU_K\n");
//        printf("dict size = %zd\n", PyDict_Size(cache_params));
//        PyObject* list = PyDict_Keys(cache_params);
//        PyObject* list0 = PyList_GetItem(list, 0);
//        printf("key unicode = %d\n", PyUnicode_Check(list0));
//        
//        list = PyDict_Values(cache_params);
//        list0 = PyList_GetItem(list, 0);
//        printf("value is long = %d\n", PyLong_Check(list0));
//        printf("real value is %ld\n", PyLong_AsLong(list0));
//        
//        
//        printf("inside = %d\n", PyDict_Contains(cache_params, PyUnicode_FromString("K")));
//        PyObject* lO = PyDict_GetItemString(cache_params, "K");
//        printf("lO is long = %d\n", PyLong_Check(lO));
        int K = (int)PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
        init_params->K = K;
        init_params->maxK = K;
        cache = LRU_K_init(cache_size, data_type, (void*)init_params);
#ifdef DEBUG
        printf("cache->K = %d, maxK = %d\n", ((struct LRU_K_params*)(cache->cache_params))->K, ((struct LRU_K_params*)(cache->cache_params))->maxK);
#endif 
        
    }
    else if (strcmp(algorithm, "AMP") == 0){
        gint threshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "pthreshold"));
        gint K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "K"));
#ifdef DEBUG
        printf("AMP K %d, threshold %d\n", K, threshold);
#endif 
        struct AMP_init_params *init_params = g_new(struct AMP_init_params, 1);
        init_params->APT            = 4;
        init_params->read_size      = 1;
        init_params->p_threshold    = threshold;
        init_params->K              = K;
        cache = AMP_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "YJC") == 0){
        double LRU_percentage = PyFloat_AS_DOUBLE(PyDict_GetItemString(cache_params, "LRU_percentage"));
        double LFU_percentage = PyFloat_AS_DOUBLE(PyDict_GetItemString(cache_params, "LFU_percentage"));
#ifdef DEBUG
        printf("LRU_percentage=%lf\n", LRU_percentage);
        printf("LFU_percentage=%lf\n", LFU_percentage);
#endif
        struct YJC_init_params *init_params = g_new(struct YJC_init_params, 1);
        init_params->LRU_percentage = LRU_percentage;
        init_params->LFU_percentage = LFU_percentage;
        init_params->clustering_hashtable = NULL;
        cache = YJC_init(cache_size, data_type, (void*)init_params);
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
        gint AMP_pthreshold = -1;
        
        gint sequential_K = -1;
        if (sequential_type != 0)
            sequential_K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_K"));
        gchar* cache_type = "unknown";
        
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            cache_type = g_strdup( PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        if (strcmp(cache_type, "unknown") == 0){
            printf("please provide cache_type\n");
            exit(1);
        }
        if (strcmp(cache_type, "AMP") == 0)
            AMP_pthreshold = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "AMP_pthreshold"));
        
        
//        temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "mining_period_type"), "utf-8", "strict"); // Owned reference
//        if (temp_bytes != NULL) {
//            mining_period_type = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
//            Py_DECREF(temp_bytes);
//        }
//        if (strcmp(mining_period_type, "unknown") == 0){
//            printf("please provide mining_period_type\n");
//            exit(1);
//        }
        
        
#ifdef DEBUG
        printf("cache type %s, max support=%d, min support %d, confidence %d, sequential %d\n",
                     cache_type, max_support, min_support, confidence, sequential_K);
#endif
        
        struct MIMIR_init_params *init_params = g_new(struct MIMIR_init_params, 1);
        init_params->max_support            = max_support;
        init_params->min_support            = min_support;
        init_params->cache_type             = cache_type;
        init_params->confidence             = confidence;
        init_params->item_set_size          = item_set_size;
        init_params->training_period        = 0;
        init_params->prefetch_list_size     = prefetch_list_size;
        init_params->block_size             = block_size;
        init_params->max_metadata_size      = max_metadata_size;
        init_params->training_period_type   = 'v';
        init_params->sequential_type        = sequential_type;
        init_params->sequential_K           = sequential_K;
        init_params->AMP_pthreshold         = AMP_pthreshold;
        init_params->cycle_time             = cycle_time;
        
        cache = MIMIR_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "MS1") == 0){
        gint max_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "max_support"));
        gint min_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "min_support"));
        gint confidence = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "confidence"));
        gint item_set_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "item_set_size"));
        gdouble mining_period = (gdouble) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "mining_period"));
        gint prefetch_list_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_list_size"));
        gint64 prefetch_table_size = PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_table_size"));
        gint sequential_type = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_type"));
        gint sequential_K = -1;
        if (sequential_type == 1)
            sequential_K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_K"));
        gchar* cache_type = "unknown";
        gchar* mining_period_type = "unknown";
        
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            cache_type = g_strdup( PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        if (strcmp(cache_type, "unknown") == 0){
            printf("please provide cache_type\n");
            exit(1);
        }
        
        temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "mining_period_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            mining_period_type = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        if (strcmp(mining_period_type, "unknown") == 0){
            printf("please provide mining_period_type\n");
            exit(1);
        }
        
        
#ifdef DEBUG
        printf("cache type %s, max support=%d, min support %d, confidence %d, mining period %lf, sequential %d\n",
               cache_type, max_support, min_support, confidence, mining_period, sequential_K);
#endif
        
        struct MS1_init_params *init_params = g_new(struct MS1_init_params, 1);
        init_params->max_support = max_support;
        init_params->min_support = min_support;
        init_params->cache_type = cache_type;
        init_params->confidence = confidence;
        init_params->item_set_size = item_set_size;
        init_params->training_period = mining_period;
        init_params->prefetch_list_size = prefetch_list_size;
        init_params->prefetch_table_size = prefetch_table_size;
        init_params->training_period_type = mining_period_type[0];
        init_params->sequential_type = sequential_type;
        init_params->sequential_K = sequential_K;
        
        cache = MS1_init(cache_size, data_type, (void*)init_params);
    }
    else if (strcmp(algorithm, "MS2") == 0){
        gint max_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "max_support"));
        gint min_support = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "min_support"));
        gint confidence = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "confidence"));
        gint item_set_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "item_set_size"));
        gdouble mining_period = (gdouble) PyFloat_AsDouble(PyDict_GetItemString(cache_params, "mining_period"));
        gint prefetch_list_size = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_list_size"));
        gint64 prefetch_table_size = PyLong_AsLong(PyDict_GetItemString(cache_params, "prefetch_table_size"));
        gint sequential_type = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_type"));
        gint sequential_K = -1;
        if (sequential_type == 1)
            sequential_K = (gint) PyLong_AsLong(PyDict_GetItemString(cache_params, "sequential_K"));
        gchar* cache_type = "unknown";
        gchar* mining_period_type = "unknown";
        
        PyObject * temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "cache_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            cache_type = g_strdup( PyBytes_AS_STRING(temp_bytes) ); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        if (strcmp(cache_type, "unknown") == 0){
            printf("please provide cache_type\n");
            exit(1);
        }
        
        temp_bytes = PyUnicode_AsEncodedString(PyDict_GetItemString(cache_params, "mining_period_type"), "utf-8", "strict"); // Owned reference
        if (temp_bytes != NULL) {
            mining_period_type = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
            Py_DECREF(temp_bytes);
        }
        if (strcmp(mining_period_type, "unknown") == 0){
            printf("please provide mining_period_type\n");
            exit(1);
        }
        
        
#ifdef DEBUG
        printf("cache type %s, max support=%d, min support %d, confidence %d, mining period %lf, sequential %d\n",
               cache_type, max_support, min_support, confidence, mining_period, sequential_K);
#endif
        
        struct MS2_init_params *init_params = g_new(struct MS2_init_params, 1);
        init_params->max_support = max_support;
        init_params->min_support = min_support;
        init_params->cache_type = cache_type;
        init_params->confidence = confidence;
        init_params->item_set_size = item_set_size;
        init_params->training_period = mining_period;
        init_params->prefetch_list_size = prefetch_list_size;
        init_params->prefetch_table_size = prefetch_table_size;
        init_params->training_period_type = mining_period_type[0];
        init_params->sequential_type = sequential_type;
        init_params->sequential_K = sequential_K;
        
        cache = MS2_init(cache_size, data_type, (void*)init_params);
    }
    else {
        printf("does not support given cache replacement algorithm: %s\n", algorithm);
        exit(1);
    }
    return cache;
}
