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

#ifdef __cplusplus
extern "C"
{
#endif


#define MAX_FMT_LEN 128


typedef struct{
    gint label_pos;                     // ordering begins with 0
    gint op_pos;
    gint real_time_pos;
    gint size_pos;
    char fmt[MAX_FMT_LEN];
}binary_init_params_t;


typedef struct{
    gint    label_pos;                  // the beginning bytes in the struct
    guint   label_len;                  // the size of label
    char    label_type;
    
    gint    op_pos;
    guint   op_len;
    char    op_type;
    
    gint    real_time_pos;
    guint   real_time_len;
    char    real_time_type;
    
    gint    size_pos;
    guint   size_len;
    char    size_type;
    
    char fmt[MAX_FMT_LEN];
    guint   num_of_fields;
}binary_params_t;



/* binary extract extracts the attribute from record, at given pos */
static inline void binary_extract(void* record, int pos, int len,
                                  char type, void* written_to){
    
    ;
    
    switch (type) {
        case 'b':
        case 'B':
        case 'c':
        case '?':
            WARNING("given type %c cannot be used for label or time\n", type);
            break;
            
        case 'h':
            *(gint16*)written_to = *(gint16*)(record + pos);
            break;
        case 'H':
            *(guint16*)written_to = *(guint16*)(record + pos);
            break;
            
        case 'i':
        case 'l':
            *(gint32*)written_to = *(gint32*)(record + pos);
            break;

        case 'I':
        case 'L':
            *(guint32*)written_to = *(guint32*)(record + pos);
            break;
            
        case 'q':
            *(gint64*)written_to = *(gint64*)(record + pos);
            break;
        case 'Q':
            *(guint64*)written_to = *(guint64*)(record + pos);
            break;
            
        case 'f':
            *(float*)written_to = *(float*)(record + pos);
            break;
            
        case 'd':
            *(double*)written_to = *(double*)(record + pos);
            break;
            
        case 's':
            strncpy((char*) written_to, (char*)(record+pos), len);
            ((char*) written_to)[len] = 0;
            break;
            
            
        default:
            ERROR("DO NOT recognize given format character: %c\n", type);
            break;
    }
}




static inline int binary_read(reader_t* reader, cache_line* cp){
    if (reader->base->offset >= reader->base->total_num * reader->base->record_size){
        cp->valid = FALSE;
        return 0;
    }
    
    binary_params_t *params = reader->reader_params;
    
    void *record = (reader->base->mapped_file + reader->base->offset);
    if (params->label_type){
        binary_extract(record, params->label_pos, params->label_len,
                       params->label_type, cp->item_p);
    }
    if (params->real_time_type){
        binary_extract(record, params->real_time_pos, params->real_time_len,
                       params->real_time_type, &(cp->real_time));
    }
    if (params->size_type){
        binary_extract(record, params->size_pos, params->size_len,
                       params->size_type, &(cp->size));
    }
    if (params->op_type){
        printf("op type %d %c\n", params->op_type, params->op_type);
        WARNING("currently op option is not supported\n");
//        binary_extract(record, params->op_pos, params->op_len,
//                       params->op_type, &(cp->op));
    }
    
    
    (reader->base->offset) += reader->base->record_size;
    return 0;
}









/* function to setup binary reader */
int binaryReader_setup(const char *const filename,
                       reader_t *const reader,
                       const binary_init_params_t *const params);



#ifdef __cplusplus
}
#endif


#endif
