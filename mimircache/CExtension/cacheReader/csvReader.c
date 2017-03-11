//
//  csvReader.c
//  mimircache 
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "csvReader.h"


static inline void csv_cb1(void *s, size_t len, void *data){
    /* call back for csv field end */
    
    reader_t* reader = (reader_t*)data;
    csv_params_t *params = reader->reader_params;
    cache_line* cp = params->cache_line_pointer;
    
    if (params->current_column_counter == params->label_column){
        // this field is the request lable (key)
        if (len >= cache_line_label_size)
            len = cache_line_label_size - 1;
        strncpy(cp->item, (char*)s, len);
        cp->item[len] = 0;
        params->already_got_cache_line = TRUE;
    }
    else if (params->current_column_counter == params->real_time_column){
        cp->real_time = (guint64) atoll((char*) s);
    }
    else if (params->current_column_counter == params->op_column){
        fprintf(stderr, "currently operation column is not supported\n");
    }
    else if (params->current_column_counter == params->size_column){
        cp->size = (size_t) atoi((char*) s);
    }
    else if (params->current_column_counter == params->traceID_column){
        cp->traceID = (unsigned char) *((char*) s);
    }
    
    params->current_column_counter ++ ;
}

static inline void csv_cb2(int c, void *data){
    /* call back for csv row end */
    
    
    reader_t* reader = (reader_t*)data;
    csv_params_t *params = reader->reader_params;
    params->current_column_counter = 1;
    
    /* move the following code to csv_cb1 after detecting label, 
     * because putting here will cause a bug when there is no new line at the 
     * end of file, then the last line will have an incorrect reference number
     */

}


csvReader_init_params* new_csvReader_init_params(gint label_column,
                                                 gint op_column,
                                                 gint real_time_column,
                                                 gint size_column,
                                                 gboolean has_header,
                                                 unsigned char delimiter,
                                                 gint traceID_column){
    
    csvReader_init_params* init_params  =   g_new0(csvReader_init_params, 1);
    init_params->label_column           =   label_column;
    init_params->op_column              =   op_column;
    init_params->real_time_column       =   real_time_column;
    init_params->size_column            =   size_column;
    init_params->has_header             =   has_header;
    init_params->delimiter              =   delimiter;
    init_params->traceID_column         =   traceID_column;
    return init_params; 
}



void csv_setup_Reader(char* file_loc, reader_t* reader, csvReader_init_params* init_params){
    unsigned char options = CSV_APPEND_NULL;
    reader->reader_params = g_new0(csv_params_t, 1);
    csv_params_t *params = reader->reader_params;

    /* passed in init_params needs to be saved within reader, to faciliate clone and free */
    reader->base->init_params = g_new(csvReader_init_params, 1);
    memcpy(reader->base->init_params, init_params, sizeof(csvReader_init_params));
    
    params->csv_parser = g_new(struct csv_parser, 1);
    if (csv_init(params->csv_parser, options) != 0) {
        fprintf(stderr, "Failed to initialize csv parser\n");
        exit(1);
    }
    
    reader->base->file = fopen(file_loc, "rb");
    if (reader->base->file == 0) {
        ERROR("Failed to open %s: %s\n", file_loc, strerror(errno));
        exit(1);
    }
        
    reader->base->type = 'c';         /* csv  */
    params->current_column_counter = 1;
    params->op_column = init_params->op_column;
    params->real_time_column = init_params->real_time_column;
    params->size_column = init_params->size_column;
    params->label_column = init_params->label_column;
    params->already_got_cache_line = FALSE;
    params->reader_end = FALSE;
    
    
    /* if we setup something here, then we must setup in the reset_reader func */
    if (init_params->delimiter){
        csv_set_delim(params->csv_parser, init_params->delimiter);
        params->delim = init_params->delimiter;
    }
    
    if (init_params->has_header){
        char *line = NULL;
        size_t len;
        len = getline(&line, &len, reader->base->file);
        free(line);
        params->has_header = init_params->has_header;
    }
}


void csv_read_one_element(reader_t* reader, cache_line* c){
    char *line = NULL;
    size_t len;
    long size = -1;
    csv_params_t *params = reader->reader_params;

    params->cache_line_pointer = c;
    params->already_got_cache_line = FALSE;

    if (params->reader_end){
        c->valid = FALSE;
        return;
    }
    while (!params->already_got_cache_line){
        size = (long) getline(&line, &len, reader->base->file);
        if (size < 0){
            break;
        }
        if ( (long)csv_parse(params->csv_parser, line, (size_t)size,
                             csv_cb1, csv_cb2, reader) != size) {
            ERROR("Error while parsing file: %s\n",
                    csv_strerror(csv_error(params->csv_parser)));
        }
        free(line);
        line = NULL;
    }
    if (!params->already_got_cache_line){       // didn't read in trace item
        if (size < 0){
            if (feof(reader->base->file) != 0){
                // end of file, last line
                csv_fini(params->csv_parser, csv_cb1, csv_cb2, reader);
                params->reader_end = TRUE;                          // got last line
                if (!params->already_got_cache_line)
                    c->valid = FALSE;
            }
            else{
                fprintf(stderr, "error in csv reader, didn't read in "
                        "cache request and file not end\n");
                exit(1);
            }
        }
        else{
            fprintf(stderr, "error in csv reader, read in file, but not "
                    "cannot parse into cache request\n");
            exit(1);
        }
    }
}


guint64 csv_skip_N_elements(reader_t* reader, guint64 N){
    /* this function skips the next N requests, 
     * on success, return N, 
     * on failure, return the number of requests skipped 
     */
    csv_params_t *params = reader->reader_params;

    csv_free(params->csv_parser);
    csv_init(params->csv_parser, CSV_APPEND_NULL);
    char *line=NULL;
    size_t len;
    guint64 i, count;
    guint64 skipped = N;
    for (i=0; i<N; i++){
        if (getline(&line, &len, reader->base->file) > 0)
            count++;
        else{
            skipped = i;
            break;
        }
        free(line);
        line = NULL;
    }
    return skipped;
}


void csv_reset_reader(reader_t* reader){
    csv_params_t *params = reader->reader_params;
    
    fseek(reader->base->file, 0L, SEEK_SET);
    
    csv_free(params->csv_parser);
    csv_init(params->csv_parser, CSV_APPEND_NULL);
    if (params->delim)
        csv_set_delim(params->csv_parser, params->delim);
    
    if (params->has_header){
        char *line=NULL;
        size_t len;
        len = getline(&line, &len, reader->base->file);
        free(line);
        line = NULL;
    }
    params->reader_end = FALSE;    
}


void csv_set_no_eof(reader_t* reader){
    csv_params_t *params = reader->reader_params;
    params->reader_end = FALSE; 
}
