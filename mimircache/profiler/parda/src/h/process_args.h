#ifndef _PROCESS_ARGS_H
#define _PROCESS_ARGS_H

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

extern int is_omp;
extern int is_mpi;
extern int is_seperate;
extern int is_binary;
extern char inputFileName[200];
extern long lines;
extern int threads;
extern int buffersize;

int process_args(int argc,char **argv);
#endif
