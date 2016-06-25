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

#include "const.h"


struct break_point{
    GArray* array;
    char mode;
    guint64 time_interval;
};



typedef struct{
    union {
        FILE* file;
        struct{
            void* p;        /* pointer to memory location, the begin position, 
                             this should not change after initilization  */
            long long offset;    /* the position the reader pointer currently at */
            int record_size;     /* the size of one record, used to locate the memory 
                                    location of next element */
        };
    };
    char type;       /* possible types: c(csv), v(vscsi), p(plain text)  */
    long long total_num;
    long long ts;           /* current timestamp, record current line, even if some 
                             * lines are not processed(skipped) */
    char file_loc[FILE_LOC_STR_SIZE];
    struct break_point* break_points;
    long long* reuse_dist;
    gint* last_access;
    guint64 max_reuse_dist;
    GQueue * best_LRU_cache_size;
    double log_base;
    double* hit_rate;
    union{
        struct{
            int vscsi_ver;        // version
        };
    };
    int ver;
    
} READER;


typedef struct{
    union{
        char str_content[cache_line_label_size];
        int int_content;
        guint64 long_content;
    };
    // maybe we can use a memory area then define different names for it? 
    // then we can access the memory area using the same name 

    char type;      // type of content can be either long(l) or char*(c), int(i)
    long long ts;   // virtual timestamp 
    long size;
    int op;
    long long real_time;
    gboolean valid;
}cache_line;



#include <vscsi_trace_format.h> 


READER* setup_reader(char* file_loc, char file_type);
void read_one_element(READER* reader, cache_line* c);
long skip_N_elements(READER* reader, long long N);
int go_back_one_line(READER* reader);
int go_back_two_lines(READER* reader);
void read_one_element_above(READER* reader, cache_line* c);

int read_one_request_all_info(READER* reader, void* storage);
guint64 read_one_timestamp(READER* reader);
void read_one_op(READER* reader, void* op);
guint64 read_one_request_size(READER* reader);


void reader_set_read_pos(READER* reader, float pos);
long long get_num_of_cache_lines(READER* reader);
void reset_reader(READER* reader);
int close_reader(READER* reader);
READER* copy_reader(READER* reader);



#endif /* reader_h */
