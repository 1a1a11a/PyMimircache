


#include "python_wrapper.h" 




struct_cache* build_cache(READER* reader, long cache_size, char* algorithm, PyObject* cache_params, long begin){

    struct_cache *cache;
    char data_type;
    if (reader->type == 'v')
        data_type= 'l';
    else
        data_type = 'c';
    
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
    else if (strcmp(algorithm, "LRU_LFU") == 0){
        double LRU_percentage = PyFloat_AS_DOUBLE(PyDict_GetItemString(cache_params, "LRU_percentage"));
        DEBUG(printf("LRU_percentage=%lf\n", LRU_percentage));
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
        DEBUG(printf("K=%d\n", K));
//        struct LRU_K_init_params *init_params = (struct LRU_K_init_params*) malloc(sizeof(struct LRU_K_init_params));
        struct LRU_K_init_params *init_params = g_new(struct LRU_K_init_params, 1);
        init_params->K = K;
        init_params->maxK = K;
        cache = LRU_K_init(cache_size, data_type, (void*)init_params);
        DEBUG(printf("cache->K = %d, maxK = %d\n", ((struct LRU_K_params*)(cache->cache_params))->K, ((struct LRU_K_params*)(cache->cache_params))->maxK));
    }
    else {
        printf("does not support given cache replacement algorithm: %s\n", algorithm);
        exit(1);
    }
    return cache;
}