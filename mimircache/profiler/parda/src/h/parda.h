#ifndef _PARDA_H
#define _PARDA_H

#include "narray.h"
#include "process_args.h"
#include "splay.h"

#include <assert.h>
#include <glib.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <unistd.h>

/*
  An implementation of Parda, a fast parallel algorithm
to compute accurate reuse distances by analysis of reference
traces.
  Qingpeng Niu
*/

#ifdef enable_mpi
#ifdef enable_omp
#define enable_hybrid
#endif
#endif

// #define enable_timing
#ifdef enable_timing
#define PTIME(cmd) cmd
#else
#define PTIME(cmd)
#endif
// #define enable_debugging
#ifdef enable_debugging
#define DEBUG(cmd) cmd
#else
#define DEBUG(cmd)
#endif

#ifdef enable_profiling
#define PROF(cmd) cmd
#else
#define PROF(cmd)
#endif

//#define DEFAULT_NBUCKETS 100000
#define DEFAULT_NBUCKETS 1000000000
#define B_OVFL   nbuckets
#define B_INF    nbuckets+1
#define SLEN 200


/* Tunable parameters */
extern unsigned long long nbuckets;
#ifdef ENABLE_PROFILING
extern char pfile[128];
extern FILE* pid_fp;
#endif
/*data structure for parda*/
typedef char HKEY[SLEN];

typedef struct end_keytime_s {
		narray_t* gkeys;
		narray_t* gtimes;
}end_keytime_t;

typedef struct processor_info_s {
	int pid,psize;
	long tstart,tlen,tend,sum;
}processor_info_t;

typedef struct program_data_s {
  GHashTable* gh;
	narray_t* ga;
	end_keytime_t ekt;
  Tree* root;
  unsigned int *histogram;
}program_data_t;

// double* classical_tree_based_stackdist(char* inputFileName, long lines);
void classical_tree_based_stackdist(char* inputFileName, long lines, long cache_size, float* return_array);
/*functions for glib*/
gboolean compare_strings(gconstpointer a, gconstpointer b);
void iterator(gpointer key, gpointer value, gpointer ekt);

/*functions for parda core*/
program_data_t parda_init(void);
void parda_input_with_filename(char* inFileName, program_data_t* pdt, long begin, long end);
void parda_input_with_textfilepointer(FILE* fp, program_data_t* pdt, long begin, long end);
void parda_input_with_binaryfilepointer(FILE* fp, program_data_t* pdt, long begin, long end);
void parda_free(program_data_t* pdt);
end_keytime_t parda_generate_end(const program_data_t* pdt);
processor_info_t parda_get_processor_info(int pid, int psize, long sum);
void parda_get_abfront(program_data_t* pdt_a, const narray_t* gb, const processor_info_t* pit_a);
int parda_get_abend(program_data_t* pdt_b, const end_keytime_t* ekt_a);
program_data_t parda_merge(program_data_t* pdt_a, program_data_t* pdt_b,
				const processor_info_t* pit_b);
/*functions for parda print*/
void parda_print_front(const program_data_t* pdt);
void parda_print_end(const end_keytime_t* ekt);
void parda_print_tree(const program_data_t* pdt);
void parda_print_hash(const program_data_t* pdt);
void parda_print(const program_data_t* pdt);
void print_iterator(gpointer key, gpointer value, gpointer ekt);
void parda_print_histogram(const unsigned* histogram);
void parda_print_histogram_to_file(const unsigned* histogram, FILE* f);

int parda_findopt(char *option, char **value, int *argc, char ***argv);
void parda_process(char* input, T tim, program_data_t* pdt);
/*functions for mpi communication*/
void show_hkey(void* data, int i, FILE* fp);
void show_T(void* data, int i, FILE* fp);
/*functions for clock*/
double rtclock(void);
void set_return_result(const unsigned* histogram, long cache_size, float* return_array, char* inputFileName);
double* parda_return_python(const unsigned* histogram); 

void get_reuse_distance(char* inputFileName, long lines, long cache_size, long* return_array);
void parda_input_with_textfilepointer_get_reuse_distance(FILE* fp, program_data_t* pdt, long begin, long end, long* return_array);
void parda_input_with_filename_get_reuse_distance(char* inFileName, program_data_t* pdt, long begin, long end, long* return_array);





/*parda inline functions*/
static inline T parda_low(int pid, int psize, T sum) {
	return (((long long)(pid))*(sum)/(psize));
}

static inline T parda_high(int pid, int psize, T sum) {
		return parda_low(pid + 1, psize, sum)-1;
}

static inline T parda_size(int pid, int psize, T sum) {
		return (parda_low(pid + 1, psize, sum)) - (parda_low(pid, psize, sum));
}

static inline T parda_owner(T index, int psize, T sum) {
		return (((long long)psize)*(index+1)-1)/sum;
}

static inline char* parda_generate_pfilename(char filename[], int pid, int psize) {
	char pfilename[256];
	char *dir_name = dirname(strdup(filename));
	char *base_name = basename(strdup(filename)); 
	// printf("inputname addr: %p, inputname: %s, dirname: %s, basename: %s\n", filename, filename, dir_name, base_name);
	sprintf(pfilename, "%s/%d_%s_p%d.txt", dir_name, psize, base_name, pid);
	// printf("generated intermediate data file: %s\n", pfilename);
	// sprintf(pfilename, "%s/%d_%s_p%d.txt", dir_name, psize, base_name, pid);
	// printf("pfilename: %s\n", pfilename);
	return strdup(pfilename);
}

static inline void process_one_access(char* input, program_data_t* pdt, const long tim) {
		int distance;
		int *lookup;
		lookup = g_hash_table_lookup(pdt->gh, input);
    //printf("gh=%p process_one\n",pdt->gh);
		// Cold start: Not in the list yet
		if (lookup == NULL) {
				char *data = strdup(input);
				pdt->root=insert(tim,pdt->root);
				long *p_data;
        narray_append_val(pdt->ga,input);
				if ( !(p_data = (long*)malloc(sizeof(long))) )
				{
						printf("no memory for p_data\n");assert(0);exit(-1);
				}
				*p_data = tim;
				g_hash_table_insert(pdt->gh, data, p_data);  // Store pointer to list element
		}
		// Hit: We've seen this data before
		else {
				char *data = strdup(input);
				pdt->root = insert((*lookup), pdt->root);
				distance = node_size(pdt->root->right);
				pdt->root = delete(*lookup, pdt->root);
				pdt->root = insert(tim, pdt->root);
				int *p_data;
				if ( !(p_data = (int*)malloc(sizeof(int)))) {
						printf("no memory for p_data\n");
            assert(0); exit(-1);
				}
				*p_data = tim;
				g_hash_table_replace(pdt->gh, data, p_data);
				// Is distance greater than the largest bucket
				if (distance > nbuckets)
						pdt->histogram[B_OVFL] += 1;
				else
						pdt->histogram[distance] += 1;
		}
}




static inline long process_one_access_get_reuse_distance(char* input, program_data_t* pdt, const long tim) {
		long distance;
		int *lookup;
		lookup = g_hash_table_lookup(pdt->gh, input);
    //printf("gh=%p process_one\n",pdt->gh);
		// Cold start: Not in the list yet
		if (lookup == NULL) {
				distance = -1;
				char *data = strdup(input);
				pdt->root=insert(tim,pdt->root);
				long *p_data;
        narray_append_val(pdt->ga,input);
				if ( !(p_data = (long*)malloc(sizeof(long))) )
				{
						printf("no memory for p_data\n");assert(0);exit(-1);
				}
				*p_data = tim;
				g_hash_table_insert(pdt->gh, data, p_data);  // Store pointer to list element
		}
		// Hit: We've seen this data before
		else {
				char *data = strdup(input);
				pdt->root = insert((*lookup), pdt->root);
				distance = node_size(pdt->root->right);
				pdt->root = delete(*lookup, pdt->root);
				pdt->root = insert(tim, pdt->root);
				int *p_data;
				if ( !(p_data = (int*)malloc(sizeof(int)))) {
						printf("no memory for p_data\n");
            assert(0); exit(-1);
				}
				*p_data = tim;
				g_hash_table_replace(pdt->gh, data, p_data);
				// Is distance greater than the largest bucket
				if (distance > nbuckets)
						pdt->histogram[B_OVFL] += 1;
				else
						pdt->histogram[distance] += 1;
		}
		return distance;
}

GArray* line_to_str_array(char* line, char* delim); 
void parda_input_with_textfilepointer_get_reuse_distance_smart(FILE* fp, program_data_t* pdt, long begin, long end, long* return_array); 
void iterator3(gpointer key, gpointer value, gpointer user_data); 
void iterator2(gpointer key, gpointer value, gpointer user_data); 
void get_reuse_distance_smart(char* inputFileName, long lines, long cache_size, long* return_array); 
void parda_input_with_filename_get_reuse_distance_smart(char* inFileName, program_data_t* pdt, long begin, long end, long* return_array); 
void process_one_access_frequent_item(gpointer data, gpointer user_data); 


#endif
