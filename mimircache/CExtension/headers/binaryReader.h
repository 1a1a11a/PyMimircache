//
//  binaryReader.h
//  mimircache
//
//  Created by Juncheng on 2/28/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//



#ifndef _BINARY_READER_
#define _BINARY_READER_

#include "reader.h"


#define MAX_FMT_LEN 128


typedef struct{
    gint label_pos;
    gint op_pos;
    gint real_time_pos;
    gint size_pos;
    char fmt[MAX_FMT_LEN];
}binary_init_params_t;


typedef struct{
    gint    label_pos;                  // the beginning bytes in the struct
    guint   label_len;                  // the size of label
    gint    op_pos;
    guint   op_len;
    gint    real_time_pos;
    guint   real_time_len;
    gint    size_pos;
    guint   size_len;
    
    char fmt[MAX_FMT_LEN];
    guint   num_of_fields;
}binary_params_t;






static inline int binary_read(reader_t* reader, cache_line* c){
  if (reader->base->offset >= reader->base->total_num * reader->base->record_size){
    c->valid = FALSE;
    return 0;
  }
  
    void *record = (reader->base->mapped_file + reader->base->offset);
    
//    c->real_time = record->ts;
//    c->size = record->len;
//    c->op = record->cmd;
//    *((guint64*)(c->item_p)) = record->lbn;
    
    (reader->base->offset) += reader->base->record_size;
    return 1;
}



int binaryReader_setup(char *filename, reader_t* reader, binary_init_params_t* params);


#endif
