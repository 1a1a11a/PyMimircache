#include "h/parda.h"
#include "h/narray.h"

// #include<omp.h>

/*
    An implementation of Parda, a fast parallel algorithm
to compute accurate reuse distances by analysis of reference
traces.
    Qingpeng Niu
*/

unsigned long long nbuckets  = DEFAULT_NBUCKETS;
// double* return_array;
void iterator(gpointer key, gpointer value, gpointer ekt) {
    HKEY temp;
    strcpy(temp, key);
    narray_append_val(((end_keytime_t*)ekt)->gkeys, temp);
    narray_append_val(((end_keytime_t*)ekt)->gtimes, value);
}

end_keytime_t parda_generate_end(const program_data_t* pdt) {
    GHashTable *gh = pdt->gh;
    end_keytime_t ekt;
    ekt.gkeys = narray_new(sizeof(HKEY), 1000);
    ekt.gtimes = narray_new(sizeof(T), 1000);
    g_hash_table_foreach(gh, (GHFunc)iterator, &ekt);
    return ekt;
}

program_data_t parda_init() {
    program_data_t pdt;
    GHashTable *gh;
    narray_t* ga = narray_new(sizeof(HKEY), 1000);
    Tree* root;
    unsigned int *histogram;
    histogram = malloc(sizeof(unsigned int) * (nbuckets+2));
    gh = g_hash_table_new_full(g_str_hash, compare_strings, free, free);
    root = NULL;
    memset(histogram, 0, (nbuckets + 2) * sizeof(unsigned int));
    pdt.ga = ga;
    pdt.gh = gh;
    pdt.root = root;
    pdt.histogram = histogram;
    return pdt;
}

gboolean compare_strings(gconstpointer a, gconstpointer b) {
    if (strcmp(a,b) == 0)
        return TRUE;
    else
        return FALSE;
}

void parda_process(char* input, T tim, program_data_t* pdt) {
    GHashTable *gh = pdt->gh;
    Tree* root = pdt->root;
    narray_t* ga = pdt->ga;
    unsigned int *histogram = pdt->histogram;
    int distance;
    T *lookup;
    lookup = g_hash_table_lookup(gh, input);
    if (lookup == NULL) {
        char* data = strdup(input);
        root = insert(tim, root);
        T *p_data;

        narray_append_val(ga, input);
        if (!(p_data = (T*)malloc(sizeof(T)))) exit(1);
        *p_data = tim;
        g_hash_table_insert(gh, data, p_data);  // Store pointer to list element
    }

    // Hit: We've seen this data before
    else {
        root = insert((*lookup), root);
        distance = node_size(root->right);
        root = delete(*lookup, root);
        root = insert(tim, root);
        int *p_data;
        if (!(p_data = (int*)malloc(sizeof(int)))) exit(1);
        *p_data = tim;
        g_hash_table_replace(gh, strdup(input), p_data);

        // Is distance greater than the largest bucket
        if (distance > nbuckets)
            histogram[B_OVFL]++;
        else
            histogram[distance]++;
    }
    pdt->root = root;
}

processor_info_t parda_get_processor_info(int pid, int psize, long sum) {
    processor_info_t pit;
    pit.tstart = parda_low(pid, psize, sum);
    pit.tend = parda_high(pid, psize, sum);
    pit.tlen = parda_size(pid, psize, sum);
    pit.sum = sum;
    pit.pid = pid;
    pit.psize = psize;
    // printf("pid:%d, psize:%ld, sum: %ld, low:%ld, end:%ld, len:%ld\n", pit.pid, pit.psize, pit.sum, pit.tstart, pit.tend, pit.tlen); 
    return pit;
}

void parda_get_abfront(program_data_t* pdt_a, const narray_t* gb, const processor_info_t* pit_a) {
    //printf("enter abfront and before for loopheihei\n");
    T tim = pit_a->tend + 1;
    GHashTable* gh = pdt_a->gh;
    narray_t* ga = pdt_a->ga;
    Tree* root = pdt_a->root;
    unsigned* histogram = pdt_a->histogram;
    int i;
    T *lookup;
    T distance;
    unsigned len = narray_get_len(gb);
    //printf("enter abfront and before for loop\n");
    for(i=0; i < len; i++, tim++) {
        HKEY entry;
        char* temp=((HKEY*)gb->data)[i];
        strcpy(entry, temp);
        lookup = g_hash_table_lookup(gh, entry);
        //printf("merge entry %s\n",entry);
        if(lookup==NULL) {
            narray_append_val (ga, entry);
            root = insert(tim, root);
        } else {
            root = insert((*lookup), root);
            distance = node_size(root->right);
            root = delete(*lookup, root);
            root = insert(tim, root);
            if (distance > nbuckets)
                histogram[B_OVFL]++;
            else
                histogram[distance]++;
        }
    }
    pdt_a->root = root;
    pdt_a->gh = gh;
}

int parda_get_abend(program_data_t* pdt_b,
        const   end_keytime_t* ekt_a ) {
    Tree* root = pdt_b->root;
    GHashTable* gh = pdt_b->gh;
    narray_t* gkeys = ekt_a->gkeys;
    narray_t* gtimes = ekt_a->gtimes;
    unsigned len = narray_get_len(gkeys);
    unsigned i;
    HKEY key;
    T tim;
    T* lookup;
    for (i = 0; i < len; i++) {
        char* temp = ((HKEY*)gkeys->data)[i];
        strcpy(key, temp);
        tim = ((T*)(gtimes->data))[i];
        lookup = g_hash_table_lookup(gh, key);
        if (lookup == NULL) {
            char* data = strdup(key);
            root = insert(tim,root);
            T *p_data;
            if ( !(p_data = (T*)malloc(sizeof(T))) ) return -1;
            *p_data = tim;
            g_hash_table_insert(gh, data, p_data);
        }
    }
    pdt_b->root = root;
    pdt_b->gh = gh;
    return 0;
}

program_data_t parda_merge(program_data_t* pdt_a, program_data_t* pdt_b,
        const processor_info_t* pit_b) {
    program_data_t pdt;
    parda_get_abfront(pdt_a, pdt_b->ga, pit_b);
    DEBUG(printf("after get abfront %d\n", pit_b->pid);)
        narray_free(pdt_b->ga);
    pdt_a->ekt = parda_generate_end(pdt_a);
    parda_get_abend(pdt_b, &pdt_a->ekt);
    narray_free(pdt_a->ekt.gkeys);
    narray_free(pdt_a->ekt.gtimes);
    pdt.ga = pdt_a->ga;
    pdt.root = pdt_b->root;
    pdt.gh = pdt_b->gh;
    pdt.histogram = pdt_a->histogram;
    int i;
    for (i = 0; i < nbuckets+2; i++)
        pdt.histogram[i] += pdt_b->histogram[i];
    free(pdt_b->histogram);
    return pdt;
}

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void classical_tree_based_stackdist(char* inputFileName, long lines, long cache_size, float* return_array) {
#ifdef enable_timing
    double ts, te, t_init, t_input, t_print, t_free;
    ts = rtclock();
#endif
    program_data_t pdt_c = parda_init();
    PTIME(te = rtclock();)
        PTIME(t_init = te - ts;)
        parda_input_with_filename(inputFileName, &pdt_c, 0, lines - 1);
    program_data_t* pdt = &pdt_c;
    pdt->histogram[B_INF] += narray_get_len(pdt->ga);
    PTIME(ts = rtclock();)
        PTIME(t_input = ts - te;)
        set_return_result(pdt->histogram, cache_size, return_array, inputFileName);
        return; 
        // return parda_return_python(pdt->histogram);
        // parda_print_histogram(pdt->histogram);
    PTIME(te = rtclock();)
        PTIME(t_print = te - ts;)
        parda_free(pdt);
    PTIME(ts = rtclock();)
        PTIME(t_free = ts - te;)
#ifdef enable_timing
        printf("seq\n");
    printf("init time is %lf\n", t_init);
    printf("input time is %lf\n", t_input);
    printf("print time is %lf\n", t_print);
    printf("free time is %lf\n", t_free);
#endif
}

void parda_input_with_filename(char* inFileName, program_data_t* pdt, long begin, long end) {
    DEBUG(printf("enter parda_input < %s from %ld to %ld\n", inFileName, begin, end);)
        FILE* fp;
    if(!is_binary) {
        fp = fopen(inFileName, "r");
        // printf("text open haha\n");
        parda_input_with_textfilepointer(fp, pdt, begin, end);
    } else {
        fp = fopen(inFileName, "rb");
        // printf("binary open haha\n");
        parda_input_with_binaryfilepointer(fp, pdt, begin, end);
    }
    fclose(fp);
}

void parda_input_with_binaryfilepointer(FILE* fp, program_data_t* pdt, long begin,long end) {
    HKEY input;
    long t, i;
    long count;
    void** buffer = (void**)malloc(buffersize * sizeof(void*));
    for (t = begin; t <= end; t += count) {
        count = fread(buffer, sizeof(void*), buffersize, fp);
        for(i=0; i < count; i++) {
            sprintf(input, "%p", buffer[i]);
            DEBUG(printf("%s %d\n",input,i+t);)
            process_one_access(input,pdt,i+t);
        }
    }
}

void parda_input_with_textfilepointer(FILE* fp, program_data_t* pdt, long begin, long end) {
    HKEY input;
    long i;
    if (begin!=0)
        for (i=0;i<begin;i++)
            fscanf(fp, "%s", input);    

    for(i = begin; i <= end; i++) {
        fscanf(fp, "%s", input);    
        // fgets(input, SLEN, fp);
        DEBUG(printf("%s %d\n", input, i);)
        process_one_access(input, pdt, i);
    }
}

void parda_free(program_data_t* pdt) {
    narray_free(pdt->ga);
    //g_hash_table_foreach(pdt->gh, free_key_value, NULL);
    g_hash_table_destroy(pdt->gh);
    free(pdt->histogram);
    freetree(pdt->root);
}


void set_return_result(const unsigned* histogram, long cache_size, float* return_array, char* inputFileName){
    int last_bucket;
    int i;
    unsigned long long sum = 0;  // For normalizing
    // unsigned long long cum = 0;  // Cumulative output
    unsigned long long miss = 0;  // infinite output
    
    char *dir_name = dirname(inputFileName);
    char output_filename[200]; 
    sprintf(output_filename, "%s/%s", dir_name, "parda_output");

    // Find the last non-zero bucket
    last_bucket = nbuckets-1;
    while (histogram[last_bucket] == 0)
        last_bucket--;
    FILE* f = fopen(output_filename, "w");
    parda_print_histogram_to_file(histogram, f);
    fclose(f);



    for (i = 0; i <= last_bucket; i++)
        sum += histogram[i];
    sum += histogram[B_OVFL];
    sum += histogram[B_INF];

    // in the returned result array, the first element is the infinite percentage 
    for (i = 0; i <= last_bucket; i++) {
        // cum += histogram[i];
        if (i<cache_size)
            return_array[i+1] = histogram[i] / (float)sum;
        else
            miss += histogram[i];
    }

    miss += histogram[B_OVFL];  
    miss += histogram[B_INF];
    return_array[0] = miss/(float)sum;
    // printf("sum: %ld, histogram[1]: %d, \n", sum, histogram[1]);
    // printf("missed: %f\n", return_array[0]);
}





// double* parda_return_python(const unsigned* histogram){
//   int last_bucket;
//   int i;
//   unsigned long long sum = 0;  // For normalizing
//   unsigned long long cum = 0;  // Cumulative output


//   // Find the last non-zero bucket
//   last_bucket = nbuckets-1;
//   while (histogram[last_bucket] == 0)
//     last_bucket--;
//   return_array = malloc(sizeof(double)* ((last_bucket+1)*2+1));
//   return_array[0] = (last_bucket+1)*2+1;

//   for (i = 0; i <= last_bucket; i++)
//     sum += histogram[i];
//   sum += histogram[B_OVFL];
//   sum += histogram[B_INF];

//   // printf("# Dist\t     Refs\t   Refs(%%)\t  Cum_Ref\tCum_Ref(%%)\n");

//   for (i = 0; i <= last_bucket; i++) {
//     cum += histogram[i];
//     // if (histogram[i]) 
//         // printf("%6d\t%9u\t%0.8lf\t%9llu\t%0.8lf\n", i, histogram[i],
//             // histogram[i] / (double)sum, cum, cum / (double)sum);
//         return_array[i+1] = histogram[i] / (double)sum;
//         return_array[last_bucket + i+4] = cum / (double)sum;
//   }

//   cum += histogram[B_OVFL];
//   // printf("#OVFL \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_OVFL], histogram[B_OVFL]/(double)sum, cum, cum/(double)sum);
//   // return_array[i+1] = histogram[B_OVFL] / (double)sum;
//   // return_array[last_bucket + i+4] = cum / (double)sum;
    

//   cum += histogram[B_INF];
//   // printf("#INF  \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_INF], histogram[B_INF]/(double)sum, cum, cum/(double)sum);
//   //printf("#INF  \t%9u\n", histogram[B_INF]);
//   // return_array[i+2] = histogram[B_INF] / (double)sum;
//   // return_array[last_bucket + i+5] = cum / (double)sum;

//   return return_array;
// }


// void end(){
//   // int * a = malloc(sizeof(int)*10);
//   // a[0] = 10000;
//   // printf("before free %p, first element: %lf\n", return_array, return_array[0]);
//   // printf("before free %p, first element: %d\n", a, a[0]);
//   free(return_array);
//   // free(a);
//   // printf("after free %p, first element: %lf\n", return_array, return_array[0]);
//   // printf("after free %p, first element: %d\n", a, a[0]);
// }


void classical_with_line_specification(char* inputFileName, 
    long lines, long cache_size, long begin_line, long end_line, float* return_array) {

    program_data_t pdt_c = parda_init();
    parda_input_with_filename(inputFileName, &pdt_c, begin_line, end_line-1);
    program_data_t* pdt = &pdt_c;
    pdt->histogram[B_INF] += narray_get_len(pdt->ga);
    set_return_result(pdt->histogram, cache_size, return_array, inputFileName);
    return; 
        // return parda_return_python(pdt->histogram);
        // parda_print_histogram(pdt->histogram);
    printf("seq\n");
}

void parda_input_with_filename_get_reuse_distance(char* inFileName, program_data_t* pdt, long begin, long end, long* return_array) {
    DEBUG(printf("enter parda_input < %s from %ld to %ld\n", inFileName, begin, end);)
        FILE* fp;
    if(!is_binary) {
        fp = fopen(inFileName, "r");
        // printf("text open haha\n");
        parda_input_with_textfilepointer_get_reuse_distance(fp, pdt, begin, end, return_array);
    } else {
        fp = fopen(inFileName, "rb");
        // printf("binary open haha\n");
        parda_input_with_binaryfilepointer(fp, pdt, begin, end);
    }
    fclose(fp);
}


void parda_input_with_textfilepointer_get_reuse_distance(FILE* fp, program_data_t* pdt, long begin, long end, long* return_array) {
    HKEY input;
    long i;
    if (begin!=0)
        for (i=0;i<begin;i++)
            fscanf(fp, "%s", input);    

    for(i = begin; i <= end; i++) {
        fscanf(fp, "%s", input);    
        // fgets(input, SLEN, fp);
        DEBUG(printf("%s %d\n", input, i);)
        return_array[i-begin] = process_one_access_get_reuse_distance(input, pdt, i);
    }
}




void get_reuse_distance(char* inputFileName, long lines, long cache_size, long* return_array) {
    program_data_t pdt_c = parda_init();
    parda_input_with_filename_get_reuse_distance(inputFileName, &pdt_c, 0, lines - 1, return_array);
    
    // program_data_t* pdt = &pdt_c;
    // pdt->histogram[B_INF] += narray_get_len(pdt->ga);


        // set_return_result(pdt->histogram, cache_size, return_array, inputFileName);
    return; 
}


void get_reuse_distance_smart(char* inputFileName, long lines, long cache_size, long* return_array) {
    program_data_t pdt_c = parda_init();
    parda_input_with_filename_get_reuse_distance_smart(inputFileName, &pdt_c, 0, lines - 1, return_array);
    
    // program_data_t* pdt = &pdt_c;
    // pdt->histogram[B_INF] += narray_get_len(pdt->ga);


        // set_return_result(pdt->histogram, cache_size, return_array, inputFileName);
    return; 
}


void parda_input_with_filename_get_reuse_distance_smart(char* inFileName, program_data_t* pdt, long begin, long end, long* return_array) {
    DEBUG(printf("enter parda_input < %s from %ld to %ld\n", inFileName, begin, end);)
        FILE* fp;
    if(!is_binary) {
        fp = fopen(inFileName, "r");
        // printf("text open haha\n");
        parda_input_with_textfilepointer_get_reuse_distance_smart(fp, pdt, begin, end, return_array);
    } else {
        fp = fopen(inFileName, "rb");
        // printf("binary open haha\n");
        parda_input_with_binaryfilepointer(fp, pdt, begin, end);
    }
    fclose(fp);
}


void parda_input_with_textfilepointer_get_reuse_distance_smart(FILE* fp, program_data_t* pdt, long begin, long end, long* return_array) {
    HKEY input;
    long i, j;
    if (begin!=0)
        for (i=0;i<begin;i++)
            fscanf(fp, "%s", input);    

    // load frequent item mining result 
    FILE* fp_mining = fopen("../Data/mining.dat", "r"); 
    GHashTable* gh_mining = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL); 
    char line[2048]; 
     
    while (fscanf(fp_mining, "%s", line) != EOF){
        GArray *ga = line_to_str_array(line, ",");

        for (i=0; i<ga->len; i++)
            printf("%s ", g_array_index(ga, char*, i));
        printf("\n");

        for (i=0; i<ga->len; i++)
            // for each element, chech whether it is in the hash table, if not, create a single list 
            // otherwise insert 
            for (j=0; j<ga->len; j++){
                if (i != j){
                    // printf("%s-%s\n", g_array_index(ga, char*, i), g_array_index(ga, char*, j));
                    GHashTable* ght = g_hash_table_lookup(gh_mining, g_array_index(ga, char*, i)); 
                    if (ght)
                        g_hash_table_add(ght, g_strdup(g_array_index(ga, char*, j))); 
                    else{
                        GHashTable* ght = g_hash_table_new(g_str_hash, g_str_equal); 
                        g_hash_table_add(ght, g_strdup(g_array_index(ga, char*, j))); 
                        g_hash_table_insert(gh_mining, g_strdup(g_array_index(ga, char*, i)), ght);  
                    }
                }
            }
        g_array_free(ga, TRUE);
    }  
    // debug 
    g_hash_table_foreach(gh_mining, (GHFunc)iterator2, NULL);

    int extra = 0; 

    for(i = begin; i <= end; i++) {
        fscanf(fp, "%s", input);    
        // fgets(input, SLEN, fp);
        DEBUG(printf("%s %d\n", input, i);)
        GHashTable* ght = g_hash_table_lookup(gh_mining, input);

        return_array[i-begin] = process_one_access_get_reuse_distance(input, pdt, i+extra);
        if (ght){
            // hash table not empty 
            GList *gl = g_hash_table_get_keys(ght); 
            GList *gl_node = gl; 
            process_one_access_get_reuse_distance(gl_node->data, pdt, i+extra);
            while ((gl_node = g_list_next(gl_node))!=NULL){
                extra ++; 
                process_one_access_get_reuse_distance(gl_node->data, pdt, i+extra);
            }
        }
    }
    g_hash_table_destroy(gh_mining);
}


// void process_one_access_frequent_item(gpointer data, gpointer user_data){
//     process_one_access_get_reuse_distance();        
// }




GArray* line_to_str_array(char* line, char* delim){
    GArray* ga = g_array_new(FALSE, FALSE, sizeof(char *)); 
    char* token = strtok(line, delim);

    while (token){
        g_array_append_val(ga, token); 
        token = strtok(NULL, delim);
    }

    return ga;
}

void iterator3(gpointer key, gpointer value, gpointer user_data) {
    printf("%s(%d) ", (char*)key, (int)strlen(key));
}


void iterator2(gpointer key, gpointer value, gpointer user_data) {
    printf("%s: ", (gchar*)key);
    // GSList *iterator_glist; 
    // for (iterator_glist = (GSList* )value; iterator_glist; iterator_glist = iterator_glist->next)
    //     printf("%s ", (gchar*)iterator_glist->data);
    g_hash_table_foreach((GHashTable*)value, (GHFunc)iterator3, NULL);
    printf("\n");
}





int main(int argc, char* argv[]){

    FILE* fp_mining = fopen("../Data/mining.dat", "r"); 
    GHashTable* gh_mining = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL); 
    char line[2048]; 
    int i=0, j=0; 
     
    while (fscanf(fp_mining, "%s", line) != EOF){
        GArray *ga = line_to_str_array(line, ",");

        for (i=0; i<ga->len; i++)
            printf("%s ", g_array_index(ga, char*, i));
        printf("\n");

        for (i=0; i<ga->len; i++)
            // for each element, chech whether it is in the hash table, if not, create a single list 
            // otherwise insert 
            for (j=0; j<ga->len; j++){
                if (i != j){
                    // printf("%s-%s\n", g_array_index(ga, char*, i), g_array_index(ga, char*, j));
                    GHashTable* ght = g_hash_table_lookup(gh_mining, g_array_index(ga, char*, i)); 
                    if (ght)
                        g_hash_table_add(ght, g_strdup(g_array_index(ga, char*, j))); 
                    else{
                        GHashTable* ght = g_hash_table_new(g_str_hash, g_str_equal); 
                        g_hash_table_add(ght, g_strdup(g_array_index(ga, char*, j))); 
                        g_hash_table_insert(gh_mining, g_strdup(g_array_index(ga, char*, i)), ght);  
                    }
    // g_hash_table_foreach(gh_mining, (GHFunc)iterator2, NULL);
    // printf("\n\n\n");
                }
            }
        g_array_free(ga, TRUE);
    }  
    g_hash_table_foreach(gh_mining, (GHFunc)iterator2, NULL);
}
