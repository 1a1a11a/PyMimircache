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
    
    READER* reader = (READER* )data;
    
    if (reader->current_column_counter == reader->label_column){
        // this field is the request lable (key)
        cache_line* cp = reader->cache_line_pointer;
        if (len >= cache_line_label_size)
            len = cache_line_label_size - 1;
        strncpy(cp->item, (char*)s, len);
        cp->item[len] = 0;
        reader->already_got_cache_line = TRUE;
        (cp->ts) ++;
    }
    else if (reader->current_column_counter == reader->real_time_column){
        cache_line* cp = reader->cache_line_pointer;
        cp->real_time = (guint64) atoll((char*) s);
    }
    else if (reader->current_column_counter == reader->op_column){
        printf("currently operation column is not supported\n");
    }
    else if (reader->current_column_counter == reader->size_column){
        cache_line* cp = reader->cache_line_pointer;
        cp->size = (size_t) atoi((char*) s);
    }
    
    reader->current_column_counter ++ ;
}

static inline void csv_cb2(int c, void *data){
    /* call back for csv row end */
    
    
    READER* reader = (READER* )data;
    reader->current_column_counter = 0;
    
    /* move the following code to csv_cb1 after detecting label, 
     * because putting here will cause a bug when there is no new line at the 
     * end of file, then the last line will have an incorrect ts 
     */

}


csvReader_init_params* new_csvReader_init_params(gint label_column, gint op_column, gint real_time_column, gint size_column, gboolean has_header, unsigned char delimiter){
    csvReader_init_params* init_params = g_new(csvReader_init_params, 1);
    init_params->label_column = label_column;
    init_params->op_column = op_column;
    init_params->real_time_column = real_time_column;
    init_params->size_column = size_column;
    init_params->has_header = has_header;
    init_params->delimiter = delimiter;
    return init_params; 
}



void csv_setup_Reader(char* file_loc, READER* reader, csvReader_init_params* init_params){
    unsigned char options = CSV_APPEND_NULL;
    
    reader->csv_parser = g_new(struct csv_parser, 1);
    if (csv_init(reader->csv_parser, options) != 0) {
        fprintf(stderr, "Failed to initialize csv parser\n");
        exit(1);
    }
    
    reader->csv_file = fopen(file_loc, "rb");
    if (reader->csv_file == 0) {
        fprintf(stderr, "Failed to open %s: %s\n", file_loc, strerror(errno));
        exit(1);
    }
    
    reader->type = 'c';         /* csv  */
    reader->data_type = 'c';    /* char */
    reader->ts = 0;             /* ts   */
    reader->current_column_counter = 0;
    reader->op_column = init_params->op_column;
    reader->real_time_column = init_params->real_time_column;
    reader->size_column = init_params->size_column;
    reader->label_column = init_params->label_column;
    reader->already_got_cache_line = FALSE;
    reader->reader_end = FALSE;
    reader->has_header = init_params->has_header;
    
    if (init_params->delimiter)
        csv_set_delim(reader->csv_parser, init_params->delimiter);
    
    if (init_params->has_header){
        char *line = NULL;
        size_t len;
        len = getline(&line, &len, reader->csv_file);
        free(line);
    }
        
    DEBUG(printf("after initialization, current_column %d, label_column: %d, real_time_column %d, size_column: %d\n", reader->current_column_counter, reader->label_column, reader->real_time_column, reader->size_column));

}


void csv_read_one_element(READER* reader, cache_line* c){
    char *line = NULL;
    size_t len;
    long size = -1;
    reader->cache_line_pointer = c;
    reader->already_got_cache_line = FALSE;

    if (reader->reader_end){
        c->valid = FALSE;
        return;
    }
    
    while (!reader->already_got_cache_line){
        size = (long) getline(&line, &len, reader->csv_file);
        if (size < 0){
            break;
        }
        if ( (long)csv_parse(reader->csv_parser, line, (size_t)size, csv_cb1, csv_cb2, reader) != size) {
            fprintf(stderr, "Error while parsing file: %s\n", csv_strerror(csv_error(reader->csv_parser)));
        }
        free(line);
        line = NULL;
    }
    if (!reader->already_got_cache_line){       // didn't read in trace item
        if (size < 0){
            if (feof(reader->csv_file) != 0){
                // end of file, last line
                csv_fini(reader->csv_parser, csv_cb1, csv_cb2, reader);
                reader->reader_end = TRUE;                          // got last line
                if (!reader->already_got_cache_line)
                    c->valid = FALSE;
            }
            else{
                fprintf(stderr, "error in csv reader, didn't read in cache request and file not end\n");
                exit(1);
            }
        }
        else{
            fprintf(stderr, "error in csv reader, read in file, but not cannot parse into cache request\n");
            exit(1);
        }
    }
}

