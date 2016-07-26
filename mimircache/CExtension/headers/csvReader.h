//
//  csvReader.h
//  mimircache
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef csvReader_h
#define csvReader_h

#include <glib.h> 
#include "reader.h"
#include "libcsv.h"
#include <errno.h>


typedef struct{
    gint label_column;
    gint op_column;
    gint real_time_column;
    gint size_column;
    gboolean has_header;
    unsigned char delimiter;
}csvReader_init_params;


csvReader_init_params* new_csvReader_init_params(gint label_column, gint op_column, gint real_time_column, gint size_column, gboolean has_header, unsigned char delimiter);

void csv_setup_Reader(char* file_loc, READER* reader, csvReader_init_params* init_params);
void csv_read_one_element(READER* reader, cache_line* c);
int csv_go_back_one_line(READER* reader);




#endif /* csvReader_h */
