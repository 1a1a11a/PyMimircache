#ifndef _PARDA_MPI_H
#define _PARDA_MPI_H

#include "parda.h"
#include <mpi.h>

narray_t* parda_recv_array(int source, int* tag, unsigned element_size);
void parda_send_array(narray_t* ga, int dest, int* tag);
unsigned* parda_mpi_merge(program_data_t* pdt, processor_info_t* pit);
void parda_mpi_free(program_data_t* pdt, unsigned* global_his);
int parda_MPI_IO_binary_input(program_data_t *pdt, char filename[], const processor_info_t* pit);
void parda_mpi_stackdist(char* inputFileName, long lines, int processors, int argc, char **argv);
#if defined(enable_omp) && defined(enable_mpi)
void parda_hybrid_stackdist(char* inputFileName, long lines, int processors, int argc, char **argv);
#endif
#endif
