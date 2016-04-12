#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "trace_format.h"

// #define enable_debug



#ifdef enable_debug 
#define DEBUG(cmd) cmd
#else 
#define DEBUG(cmd) 
#endif 


int setup(char *argv, void** mem, int* ver, int* delta, int* num_of_rec)
{
	struct stat st;	
	char *filename = NULL;
	int f = -1;
	void *memT = NULL;
	// void *mem = NULL;
	// int done = 0;
	// int delta = 0;
	// void (*fptr)(FILE *, void *) = NULL;
	
	
	filename = argv;
	if ( (f = open (filename, O_RDONLY)) < 0)
	{
		fprintf (stderr, "Unable to open '%s'\n", filename);
		return -1;
	}

	if ( (fstat (f, &st)) < 0)	
	{
		close (f);
		fprintf (stderr, "Unable to fstat '%s'\n", filename);
		return -1;
	}

	/* Version 2 records are the bigger of the two */
	if (st.st_size < sizeof(trace_v2_record_t) * MAX_TEST)
	{
		close (f);
		fprintf (stderr, "File too small, unable to read header.\n");
		return -1;
	}

	if ( (memT = (mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, f, 0))) == MAP_FAILED)
	{
		close (f);
		fprintf (stderr, "Unable to allocate %lu bytes of memory\n", st.st_size);
		return -1;
	}
	*mem = memT;
	switch (test_vscsi_version (*mem))
	{
		case VSCSI1:
			fprintf (stderr, "Detected VSCSI1 format.\n");
			// fptr = print_trace_v1;
			*delta = sizeof(trace_v1_record_t);
			*ver = VSCSI1; 
			*num_of_rec = st.st_size/(*delta);
			break;

		case VSCSI2:
			fprintf (stderr, "Detected VSCSI2 format.\n");
			// fptr = print_trace_v2;
			*delta = sizeof(trace_v2_record_t);
			*ver = VSCSI2;
			*num_of_rec = st.st_size/(*delta);
			break;
		
		case UNKNOWN:
		default:
			fprintf (stderr, "Trace format unrecognized.\n");
			return -1;
	}
	close(f);
	return 0;
}

// long read_trace(void** mem, int* ver, int* delta){
// 	// long (*fptr)(void *) = NULL;
// 	void* ptr; 
// 	switch (*ver)
// 	{
// 		case VSCSI1:
// 			ptr = (trace_v1_record_t*);
// 			break;

// 		case VSCSI2:	
// 			ptr = (trace_v2_record_t*);
// 			break;
// 	}
// 	// printf("mem: %p\n", *mem);

// 	long lbn = fptr(*mem);
// 	*mem += *delta;
// 	DEBUG(printf("mem: %p\n", *mem);)

// 	return lbn;
// }

int read_trace2(void** mem, int* ver, int size, long ts[], int len[], long lbn[], int cmd[]){
	int (*fptr)(void**, int, long*, int*, long*, int*, int) = NULL;
	int delta;
	switch (*ver)
	{
		case VSCSI1:
			fptr = set_return_value_v1;
			delta = sizeof(trace_v1_record_t);
			break;

		case VSCSI2:	
			fptr = set_return_value_v2;
			delta = sizeof(trace_v2_record_t);
			break;
	}
	// printf("mem: %p\n", *mem);

	// long lbn = fptr(*mem);
	return fptr(mem, size, ts, len, lbn, cmd, delta);
}



int finish(void* mem, int size){

	munmap (mem, size);
	return 0;
}



// int main(int argc, char** argv){
// 	void ** mem = NULL; 
// 	int ver; 
// 	int delta; 
// 	int num_of_rec; 
// 	setup(argv[1], mem, &ver, &delta, &num_of_rec);
// 	int i;
// 	void* mem_original = *mem;
// 	for (i=0;i<10;i++)
// 		printf("%ld\n", read_trace(mem, &ver, &delta));
// 	finish(mem_original, (delta)*(num_of_rec));
// 	return 0;
// }
