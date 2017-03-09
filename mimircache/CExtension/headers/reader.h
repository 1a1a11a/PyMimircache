//
//  reader.h
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef reader_h
#define reader_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <glib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>


#include "const.h"
#include "libcsv.h"


#define LINE_ENDING '\n'
#define BINARY_FMT_MAX_LEN 32


// trace type
#define CSV 'c'
#define PLAIN 'p'
#define VSCSI 'v'
#define BINARY 'b'



// data type
#define DT_NUMBER 'l'
#define DT_STRING 'c'



typedef struct break_point{
    GArray* array;
    char mode;
    guint64 time_interval;
}break_point_t;


typedef struct{
    gpointer item_p;
    char item[cache_line_label_size];
    char type;                              /* type of content can be either guint64(l) or char*(c) */
    guint64 ts;                             /* deprecated, should not use, virtual timestamp */
    size_t size;
    int op;
    guint64 real_time;
    gboolean valid;
    
    unsigned char traceID;                           /* this is for mixed trace */
    void *content;                                   /* the content of page/request */
    guint size_of_content;                           /* the size of mem area content points to */
    
    
}cache_line;


// declare reader struct 
struct reader;

typedef struct reader_base{
    
    char type;                              /* possible types: c(csv), v(vscsi),
                                             * p(plain text), b(binaryReader)  */
    char data_type;                         /* possible types: l(guint64), c(char*) */
    
    FILE* file;
    char file_loc[FILE_LOC_STR_SIZE];
    void* init_params;
    
    void* mapped_file;                      /* mmap the file, this should not change during runtime
                                             * offset in the file is changed using offset */
    guint64 offset;
    size_t record_size;                     /* the size of one record, used to
                                             * locate the memory location of next element,
                                             * used in vscsiReaser and binaryReader */
    
    gint64 total_num;                       /* number of records */
//    guint64 ref_num;                        /* current reference number, virtual timestamp,
//                                             * in other words, it is the order of current request
//                                             * in the trace */
    
    
    gint ver;
    
    void* params;                           /* currently not used */
    
    
//    void    (*read_one_element)(struct reader*, cache_line*);
//    
//    guint64 (*skip_N_elements)(struct reader*, guint64);
//    
//    int     (*go_back_one_line)(struct reader*);
//    
//    int     (*go_back_two_lines)(struct reader*);
//    
//    void    (*read_one_element_above)(struct reader*, cache_line*);
//    
//    void    (*reader_set_read_pos)(struct reader*, double);
//    
//    guint64 (*get_num_of_cache_lines)(struct reader*);
//    
//    void    (*reset_reader)(struct reader*);
//    
//    int     (*close_reader)(struct reader*);
//
//    int     (*close_reader_unique)(struct reader*);
//    
//    struct reader* (*clone_reader)(struct reader*);
//    
//    void    (*set_no_eof)(struct reader*); 
    
    
} reader_base_t;


typedef struct reader_data_unique{

    double* hit_rate;
    double log_base;

}reader_data_unique_t;


typedef struct reader_data_share{
    break_point_t *break_points;
    gint64* reuse_dist;
    gint64 max_reuse_dist;
    gint* last_access;
    
}reader_data_share_t;



//typedef struct reader_base reader_base_t;
typedef struct reader{
    struct reader_base* base;
    struct reader_data_unique* udata;
    struct reader_data_share* sdata; 
    void* reader_params;
}reader_t;









reader_t* setup_reader(char* file_loc, char file_type, char data_type, void* setup_params);
void read_one_element(reader_t* reader, cache_line* c);
guint64 skip_N_elements(reader_t* reader, guint64 N);
int go_back_one_line(reader_t* reader);
int go_back_two_lines(reader_t* reader);
void read_one_element_above(reader_t* reader, cache_line* c);

int read_one_request_all_info(reader_t* reader, void* storage);
guint64 read_one_timestamp(reader_t* reader);
void read_one_op(reader_t* reader, void* op);
guint64 read_one_request_size(reader_t* reader);


void reader_set_read_pos(reader_t* reader, double pos);
guint64 get_num_of_cache_lines(reader_t* reader);
void reset_reader(reader_t* reader);
int close_reader(reader_t* reader);
int close_reader_unique(reader_t* reader);
reader_t* clone_reader(reader_t* reader);
void set_no_eof(reader_t* reader);


cache_line* new_cacheline(void);
void destroy_cacheline(cache_line* cp);



#endif /* reader_h */
