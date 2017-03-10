//
//  reader.c
//  mimircache
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "reader.h"
#include "vscsiReader.h"
#include "binaryReader.h"
#include "csvReader.h"

/* when label/LBA is number, we plus one to it, to avoid cases of block 0 */



reader_t* setup_reader(char* file_loc, char file_type, char data_type, void* setup_params){
    /* setup the reader struct for reading trace
     file_type: c: csv, v: vscsi, p: plain text, b: binary
     data_type: l: guint64, c: string
     Return value: a pointer to READER struct, the returned reader
     needs to be explicitly closed by calling close_reader */
    
    reader_t* reader = g_new0(reader_t, 1);
    reader->base = g_new0(reader_base_t, 1);
    reader->sdata = g_new0(reader_data_share_t, 1);
    reader->udata = g_new0(reader_data_unique_t, 1);
    
    reader->sdata->break_points = NULL;
    reader->sdata->last_access = NULL;
    reader->sdata->reuse_dist = NULL;
    reader->sdata->max_reuse_dist = 0;
    
    reader->udata->hit_rate = NULL;
    
    reader->base->total_num = -1;
    reader->base->data_type = data_type;
    reader->base->init_params = NULL;
//    reader->base->ref_num = 0; 
//    reader->base->setup_reader = setup_reader;
//    reader->base->read_one_element = read_one_element;
//    reader->base->skip_N_elements = skip_N_elements;
//    reader->base->go_back_one_line = go_back_one_line;
//    reader->base->go_back_two_lines = go_back_two_lines;
//    reader->base->read_one_element_above = read_one_element_above;
//    reader->base->reader_set_read_pos = reader_set_read_pos;
//    reader->base->get_num_of_cache_lines = get_num_of_cache_lines;
//    reader->base->reset_reader = reset_reader;
//    reader->base->close_reader = close_reader;
//    reader->base->close_reader_unique = close_reader_unique;
//    reader->base->clone_reader = clone_reader;
//    reader->base->set_no_eof = set_no_eof;
    
    
    
    if (strlen(file_loc) > FILE_LOC_STR_SIZE-1){
        ERROR("file name/path is too long(>%d), "
                "please use a shorter name\n", FILE_LOC_STR_SIZE);
        exit(1);
    }
    else{
        strcpy(reader->base->file_loc, file_loc);
    }
    
    switch (file_type) {
        case 'c':
            csv_setup_Reader(file_loc, reader, setup_params);
            break;
        case 'p':
            reader->base->type = 'p';
            reader->base->file = fopen(file_loc, "r");
            if (reader->base->file == 0){
                perror("open trace file failed\n");
                exit(1);
            }
            break;
        case 'v':
            vscsi_setup(file_loc, reader);
            break;
        case 'b':
            binaryReader_setup(file_loc, reader, setup_params);
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    return reader;
}

void read_one_element(reader_t* reader, cache_line* c){
    /* read one cache line from reader,
     and store it in the pre-allocated cache_line c, current given
     size for the element(label) is 128 bytes(cache_line_label_size).
     */
    c->ts ++;
    switch (reader->base->type) {
        case 'c':
            csv_read_one_element(reader, c);
            break;
        case 'p':
            
            if (fscanf(reader->base->file, "%s", c->item) == EOF)
                //should change to the following to avoid buffer overflow
                //              if (fgets(c->item, cache_line_label_size, reader->base->file))
                c->valid = FALSE;
            else {
                if (strlen(c->item)==2 && c->item[0] == LINE_ENDING && c->item[1] == '\0')
                    return read_one_element(reader, c);
            }
            break;
        case 'v':
            vscsi_read(reader, c);
            *(guint64*) (c->item_p) = *(guint64*)(c->item_p) + 1;
            break;
        case 'b':
            binary_read(reader, c);
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    if (reader->base->data_type == 'l' &&
        (reader->base->type == 'c' || reader->base->type == 'p')){
        guint64 n = atoll(c->item);
        *(guint64*) (c->item_p) = n; 
    }
}

int go_back_one_line(reader_t* reader){
    /* go back one cache line
     return 0 on successful, non-zero otherwise
     */
    FILE *file = NULL;
    switch (reader->base->type) {
        case 'c':
        case 'p':
            file = reader->base->file;
            int r;
            gboolean after_empty_lines=FALSE;         // flag for jump over multiple empty lines at the end of file
            if (fseek(file, -2L, SEEK_CUR)!=0)
                return 1;
            char c = getc(file);
            if (c != LINE_ENDING && c!=' ' && c!='\t' && c!=',' && c!='.')
                after_empty_lines = TRUE;
            else{
                // this might happen if the length of line is 1(excluding \n)
                char c2 = getc(file);
                if (c2 != LINE_ENDING && c2!=' ' && c2!='\t' && c2!=',' && c2!='.')
                    after_empty_lines = TRUE;
                fseek(file, -1L, SEEK_CUR);
            }
            while( c != LINE_ENDING || !after_empty_lines){
                if (( r= fseek(file, -2L, SEEK_CUR)) != 0)
                    return 1;
                c = getc(file);
                if (!after_empty_lines && c != LINE_ENDING && c!=' ' && c!='\t' && c!=',' && c!='.')
                    after_empty_lines = TRUE;
            }
            
            return 0;
            
        case 'b':
        case 'v':
            if (reader->base->offset >= reader->base->record_size)
                reader->base->offset -= (reader->base->record_size);
            else
                return -1;
            return 0;
            
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
    }
}



int go_back_two_lines(reader_t* reader){
    /* go back two cache lines
     return 0 on successful, non-zero otherwise
     */
    switch (reader->base->type) {
        case 'c':
            if (go_back_one_line(reader)==0){
                go_back_one_line(reader);
                return 0;
            }
            else
                return 1;
        case 'p':
            ;
            int r;
            if ( (r=fseek(reader->base->file, -2L, SEEK_CUR)) != 0){
                return r;
            }
            
            while( getc(reader->base->file) != LINE_ENDING )
                if ( (r= fseek(reader->base->file, -2L, SEEK_CUR)) != 0){
                    return r;
                }
            fseek(reader->base->file, -2L, SEEK_CUR);
            while( getc(reader->base->file) != LINE_ENDING )
                if ( (r=fseek(reader->base->file, -2L, SEEK_CUR)) != 0){
                    fseek(reader->base->file, -1L, SEEK_CUR);
                    break;
                }
            return 0;
            // the following will have problem when each line is length 1
            //            if (go_back_one_line(reader)==0){
            //                return go_back_one_line(reader);
            ////                return 0;
            //            }
            //            else
            //                return 1;
        case 'b':
        case 'v':
            if (reader->base->offset >= (reader->base->record_size * 2))
                reader->base->offset -= (reader->base->record_size)*2;
            else
                return -1;
            return 0;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
    }
}



void read_one_element_above(reader_t* reader, cache_line* c){
    /* read one cache line from reader precede current position,
     and store it in the pre-allocated cache_line c, current given
     size for the element(label) is 128 bytes(cache_line_label_size).
     after reading the new position is at the beginning of readed cache line
     in other words, this method is called for reading from end to beginngng
     */
    if (go_back_two_lines(reader) == 0)
        read_one_element(reader, c);
    else{
        c->valid = FALSE;
    }
    
    
}



guint64 skip_N_elements(reader_t* reader, guint64 N){
    /* skip the next following N elements,
     Return value: the number of elements that are actually skipped,
     this will differ from N when it reaches the end of file
     */
    int i;
    guint64 count=0;
    char temp[cache_line_label_size];
    
    switch (reader->base->type) {
        case 'c':
            count = csv_skip_N_elements(reader, N);
            break;
        case 'p':
            for (i=0; i<N; i++)
                if (fscanf(reader->base->file, "%s", temp)!=EOF)
                    count++;
                else
                    break;
            break;
        case 'b':
        case 'v':
            if (reader->base->offset + N * reader->base->record_size <=
                reader->base->total_num * reader->base->record_size) {
                reader->base->offset = reader->base->offset + N * reader->base->record_size;
                count = N;
            }
            else{
                count = (guint64) ((reader->base->total_num * reader->base->record_size -
                                    reader->base->offset) / reader->base->record_size);
                reader->base->offset = reader->base->total_num * reader->base->record_size;
                WARNING("required to skip %lu requests, but only %lu requests left\n",
                        N, count);
            }
            
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
//    reader->base->ref_num += count;
    return count;
}

void reset_reader(reader_t* reader){
    /* rewind the reader back to beginning */
    
    switch (reader->base->type) {
        case 'c':
            csv_reset_reader(reader);
            break;
        case 'p':
            fseek(reader->base->file, 0L, SEEK_SET);
            break;
        case 'b':
        case 'v':
            reader->base->offset = 0;
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
}


void reader_set_read_pos(reader_t* reader, double pos){
    /* jump to given postion, like 1/3, or 1/2 and so on
     * reference number will NOT change in the function! .
     * due to above property, this function is deemed as deprecated.
     */
    switch (reader->base->type) {
        case 'c':
        case 'p':
            ;
            struct stat statbuf;
            if (fstat(fileno(reader->base->file), &statbuf) < 0){
                ERROR("fstat error");
                exit(0);
            }
            long long fsize = (long long)statbuf.st_size;
            fseek(reader->base->file, (long)(fsize*pos), SEEK_SET);
            char c = getc(reader->base->file);
            while (c!=LINE_ENDING && c!=EOF)
                c = getc(reader->base->file);
            break;
            
        case 'b':
        case 'v':
            reader->base->offset = (guint64)reader->base->record_size *
                                    (gint64)(reader->base->total_num * pos);
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    
}


guint64 get_num_of_cache_lines(reader_t* reader){
    
#define BUFFER_SIZE 1024*1024
    if (reader->base->total_num !=0 && reader->base->total_num != -1)
        return reader->base->total_num;
    
    guint64 num_of_lines = 0;
    char temp[BUFFER_SIZE+1];       // 1MB buffer
    int fd = 0, i=0;
    char last_char = 0;
    long size;
    reset_reader(reader);
    
    switch (reader->base->type) {
            // why not use getline here? 0308
        case 'c':
            /* same as plain text, except when has_header, it needs to reduce by 1  */
        case 'p':
            fd = fileno(reader->base->file);
            lseek(fd, 0L, SEEK_SET);
            while ((size=read(fd, (void*)temp, BUFFER_SIZE)) != 0){
                if (temp[0] == LINE_ENDING && last_char != LINE_ENDING)
                    num_of_lines++;
                for (i=1;i<size;i++)
                    if (temp[i] == LINE_ENDING && temp[i-1] != LINE_ENDING)
                        num_of_lines++;
                last_char = temp[size-1];
            }
            if (last_char!=LINE_ENDING){
                num_of_lines++;
            }
            break;
        case 'b':
        case 'v':
            return reader->base->total_num;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    if (reader->base->type == 'c' && ((csv_params_t*)reader->reader_params)->has_header)
        num_of_lines --;
    reader->base->total_num = num_of_lines;
    reset_reader(reader);
    return num_of_lines;
}


reader_t* clone_reader(reader_t* reader_in){
    /* this function clone the given reader to give an exactly same reader */
    
    reader_t* reader = setup_reader(reader_in->base->file_loc, reader_in->base->type,
                                    reader_in->base->data_type, reader_in->base->init_params);
//    FILE *temp_file = reader->base->file;
//    memcpy(reader->base, reader_in->base, sizeof(reader_base_t));
//    reader->base->file = temp_file;
    memcpy(reader->sdata, reader_in->sdata, sizeof(reader_data_share_t));
    
    if (reader->base->type == CSV){
        csv_params_t* params = reader->reader_params;
        csv_params_t* params_in = reader_in->reader_params;
        
        fseek(reader->base->file, ftell(reader_in->base->file), SEEK_SET);
        memcpy(params->csv_parser, params_in->csv_parser, sizeof(struct csv_parser));
    }
    else if (reader->base->type == PLAIN){
        fseek(reader->base->file, ftell(reader_in->base->file), SEEK_SET);
    }
    else if (reader->base->type == VSCSI || reader->base->type == BINARY){
        reader->base->offset = reader_in->base->offset;
    }
        
    else{
        ERROR("cannot recognize reader type, given reader type: %c\n",
              reader->base->type);
        exit(1);
    }
    return reader;
}




int close_reader(reader_t* reader){
    /* close the file in the reader or unmmap the memory in the file
     then free the memory of reader object
     Return value: Upon successful completion 0 is returned.
     Otherwise, EOF is returned and the global variable errno is set to
     indicate the error.  In either case no further
     access to the stream is possible.*/
    
    switch (reader->base->type) {
        case 'c':
            ;
            csv_params_t *params = reader->reader_params;
            fclose(reader->base->file);
            csv_free(params->csv_parser);
            g_free(params->csv_parser);
            break;
        case 'p':
            fclose(reader->base->file);
            break;
        case 'b':
        case 'v':
            munmap (reader->base->mapped_file, reader->base->total_num * reader->base->record_size);
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
    }
    
    if (reader->base->init_params)
        g_free(reader->base->init_params);
    
    if (reader->reader_params)
        g_free(reader->reader_params);

    if (reader->sdata){
        if (reader->sdata->last_access){
            g_free(reader->sdata->last_access);
        }
        
        if (reader->sdata->reuse_dist){
            g_free(reader->sdata->reuse_dist);
        }
        
        if (reader->sdata->break_points){
            g_array_free(reader->sdata->break_points->array, TRUE);
            g_free(reader->sdata->break_points);
        }
    }
    if (reader->udata){
        if (reader->udata->hit_rate)
            g_free(reader->udata->hit_rate);
    }

    g_free(reader->base);
    g_free(reader->udata);
    g_free(reader->sdata);
    g_free(reader);
    return 0;
}


int close_reader_unique(reader_t* reader){
    /* close the file in the reader or unmmap the memory in the file
     then free the memory of reader object
     Return value: Upon successful completion 0 is returned.
     Otherwise, EOF is returned and the global variable errno is set to
     indicate the error.  In either case no further
     access to the stream is possible.*/
    
    switch (reader->base->type) {
        case 'c':
            ;
            csv_params_t *params = reader->reader_params;
            fclose(reader->base->file);
            csv_free(params->csv_parser);
            g_free(params->csv_parser);
            break;
        case 'p':
            fclose(reader->base->file);
            break;
        case 'b':
        case 'v':
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
    }
    
    if (reader->base->init_params)
        g_free(reader->base->init_params);
    
    if (reader->reader_params)
        g_free(reader->reader_params);

    if (reader->udata){
        if (reader->udata->hit_rate)
            g_free(reader->udata->hit_rate);
    }
    
    g_free(reader->base);
    g_free(reader->sdata);
    g_free(reader->udata);
    
    g_free(reader);
    return 0;
}





int read_one_request_all_info(reader_t* reader, void* storage){
    /* read one cache line from reader,
     and store it in the pre-allocated memory pointed at storage.
     return 1 when finished or error, otherwise, return 0.
     */
    switch (reader->base->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':
            if (fscanf(reader->base->file, "%s", (char*)storage) == EOF)
                return 1;
            else {
                if (strlen((char*)storage)==2 && ((char*)storage)[0] == LINE_ENDING && ((char*)storage)[1] == '\0')
                    return read_one_request_all_info(reader, storage);
                return 0;
            }
            break;
        case 'b':
        case 'v':
            printf("currently v/b reader is not supported yet\n");
            exit(1);
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    return 0;
}

void set_no_eof(reader_t* reader){
    // remove eof flag for reader 
    switch (reader->base->type) {
        case 'c':
            csv_set_no_eof(reader);
            break;
        case 'p':
        case 'b':
        case 'v':
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
}



guint64 read_one_timestamp(reader_t* reader){
    return 0;
}


void read_one_op(reader_t* reader, void* op){
    ;
}

guint64 read_one_request_size(reader_t* reader){
    return 0;
}

cache_line* new_cacheline(){
    cache_line* cp = g_new0(cache_line, 1);
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    cp->item_p = (gpointer)cp->item;
    cp->ts = 0; 
    
    return cp;
}

void destroy_cacheline(cache_line* cp){
    if (cp->content)
        g_free(cp->content); 
    g_free(cp);
}
