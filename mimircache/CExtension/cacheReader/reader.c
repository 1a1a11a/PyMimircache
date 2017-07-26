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

/* special case in ascii reader 
 * multiple blank line in the middle 
 * each line has only one character 
 * multiple or non blank line at the end of file 
 * csv header when go back one line 
 * ending in go back one line
 */



#ifdef __cplusplus
extern "C"
{
#endif


reader_t* setup_reader(const char* const file_loc,
                       const char file_type,
                       const char data_type,
                       const int block_unit_size,
                       const int disk_sector_size,
                       const void* const setup_params){
    /* setup the reader struct for reading trace
     file_type: c: csv, v: vscsi, p: plain text, b: binary
     data_type: l: guint64, c: string
     Return value: a pointer to READER struct, the returned reader
     needs to be explicitly closed by calling close_reader */
    
    int fd;
    struct stat st;
    reader_t *const reader = g_new0(reader_t, 1);
    reader->base = g_new0(reader_base_t, 1);
    reader->sdata = g_new0(reader_data_share_t, 1);
    reader->udata = g_new0(reader_data_unique_t, 1);
    
    reader->sdata->break_points = NULL;
    reader->sdata->last_access = NULL;
    reader->sdata->reuse_dist = NULL;
    reader->sdata->max_reuse_dist = 0;
    
    reader->udata->hit_rate = NULL;
    
    reader->base->total_num = -1;
    reader->base->block_unit_size = block_unit_size;
    reader->base->disk_sector_size = disk_sector_size; 
    reader->base->data_type = data_type;
    reader->base->init_params = NULL;
    reader->base->offset = 0;
    
    
    if (strlen(file_loc) > FILE_LOC_STR_SIZE-1){
        ERROR("file name/path is too long(>%d), "
                "please use a shorter name\n", FILE_LOC_STR_SIZE);
        exit(1);
    }
    else{
        strcpy(reader->base->file_loc, file_loc);
    }
    
    
    // set up mmap region
    if ( (fd = open (file_loc, O_RDONLY)) < 0){
        ERROR("Unable to open '%s'\n", file_loc);
        exit(1);
    }
    
    if ( (fstat (fd, &st)) < 0){
        close (fd);
        ERROR("Unable to fstat '%s'\n", file_loc);
        exit(1);
    }
    
    if ( (reader->base->mapped_file = mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED){
        close (fd);
        reader->base->mapped_file = NULL;
        ERROR("Unable to allocate %llu bytes of memory, %s\n", (unsigned long long) st.st_size,
              strerror(errno));
        abort();
    }
    
    reader->base->file_size = st.st_size; 
    
    
    switch (file_type) {
        case CSV:
            csv_setup_Reader(file_loc, reader, setup_params);
            break;
        case PLAIN:
            reader->base->type = 'p';
            reader->base->file = fopen(file_loc, "r");
            if (reader->base->file == 0){
                ERROR("open trace file %s failed: %s\n", file_loc, strerror(errno));
                exit(1);
            }
            break;
        case VSCSI:
            vscsi_setup(file_loc, reader);
            break;
        case BINARY:
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

void read_one_element(reader_t *const reader, cache_line_t *const c){
    /* read one cache line from reader,
     and store it in the pre-allocated cache_line c, current given
     size for the element(label) is 128 bytes(cache_line_label_size).
     */
    c->ts ++;
    char *line_end = NULL;
    long line_len;
    switch (reader->base->type) {
        case CSV:
            csv_read_one_element(reader, c);
            break;
        case PLAIN:
            if (reader->base->offset == reader->base->file_size-1){
                c->valid = FALSE;
                break;
            }
            
            find_line_ending(reader, &line_end, &line_len);
            strncpy(c->item, reader->base->mapped_file+reader->base->offset, line_len);
            c->item[line_len] = 0;
            reader->base->offset = (void*)line_end - reader->base->mapped_file;
            break;
        case VSCSI:
            vscsi_read(reader, c);
//            *(guint64*) (c->item_p) += 1;
            break;
        case BINARY:
            binary_read(reader, c);
//            *(guint64*) (c->item_p) += 1; 
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    if (reader->base->data_type == 'l' &&
        (reader->base->type == 'c' || reader->base->type == 'p')){
        *(guint64*)(c->item_p) = atoll(c->item);
//        guint64 n = atoll(c->item);
//        *(guint64*) (c->item_p) = n + 1; // this is to avoid the situation when block number is 0
    }
}

int go_back_one_line(reader_t *const reader){
    /* go back one cache line
     return 0 on successful, non-zero otherwise
     */
    switch (reader->base->type) {
        case CSV:
        case PLAIN:
            if (reader->base->offset == 0)
                return 1;
            
            // use last record size to save loop
            const char * cp = reader->base->mapped_file + reader->base->offset;
            if (reader->base->record_size){
                cp -= reader->base->record_size - 1;
            }
            else{
                // no record size, can only happen when it is the last line
                cp --;
                // find the first current line ending
                while (*cp == FILE_LF || *cp == FILE_CR){
                    cp--;
                    if ((void*)cp < reader->base->mapped_file)
                        return 1;
                }
            }
            // now points to either end of current line letters/non-LFCR
            // or points to somewhere after the beginning of current line beginning
            // find the first character of current line
            while ( (void*)cp > reader->base->mapped_file &&
                   *cp != FILE_LF && *cp != FILE_CR){
                cp--;
            }
            if ((void*)cp != reader->base->mapped_file)
                cp ++; // jump over LFCR
            
            if ((void*)cp < reader->base->mapped_file){
                ERROR("current pointer points before mapped file\n");
                exit(1);
            }
            // now cp points to the LFCR before the line that should be read
            reader->base->offset = (void*)cp - reader->base->mapped_file; 
            
            return 0;
            
        case BINARY:
        case VSCSI:
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



int go_back_two_lines(reader_t *const reader){
    /* go back two cache lines
     return 0 on successful, non-zero otherwise
     */
    switch (reader->base->type) {
        case CSV:
        case PLAIN:
            if (go_back_one_line(reader)==0){
                reader->base->record_size = 0; 
                return go_back_one_line(reader);
            }
            else
                return 1;
        case BINARY:
        case VSCSI:
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



void read_one_element_above(reader_t *const reader, cache_line_t* c){
    /* read one cache line from reader precede current position,
     * in other words, read the line above current line, 
     * and currently file points to either the end of current line or 
     * beginning of next line.
     then store it in the pre-allocated cache_line c, current given
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



guint64 skip_N_elements(reader_t *const reader, const guint64 N){
    /* skip the next following N elements,
     Return value: the number of elements that are actually skipped,
     this will differ from N when it reaches the end of file
     */
    guint64 count=N;
    
    switch (reader->base->type) {
        case CSV:
            csv_skip_N_elements(reader, N);
        case PLAIN:
            ;
            char *line_end = NULL;
            guint64 i;
            gboolean end = FALSE;
            long line_len;
            for (i=0; i<N; i++){
                end = find_line_ending(reader, &line_end, &line_len);
                reader->base->offset = (void*)line_end - reader->base->mapped_file;
                if (end) {
                    if (reader->base->type == 'c'){
                        csv_params_t *params = reader->reader_params;
                        params->reader_end = TRUE;
                    }
                    count = i + 1;
                    break;
                }
            }

            break;
        case BINARY:
        case VSCSI:
            if (reader->base->offset + N * reader->base->record_size
                <= reader->base->file_size) {
                reader->base->offset = reader->base->offset + N * reader->base->record_size;
            }
            else{
                count = (guint64) ((reader->base->file_size - reader->base->offset)
                                   / reader->base->record_size);
                reader->base->offset = reader->base->file_size;
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
    return count;
}

void reset_reader(reader_t *const reader){
    /* rewind the reader back to beginning */
    reader->base->offset = 0;
    switch (reader->base->type) {
        case CSV:
            csv_reset_reader(reader);
        case PLAIN:
        case BINARY:
        case VSCSI:
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
}


void reader_set_read_pos(reader_t *const reader, const double pos){
    /* jump to given postion, like 1/3, or 1/2 and so on
     * reference number will NOT change in the function! .
     * due to above property, this function is deemed as deprecated.
     */
    reader->base->offset = (long)(reader->base->file_size * pos);
    if (reader->base->type == 'c' || reader->base->type == 'p'){
        reader->base->record_size = 0;
        /* for plain and csv file, if it points to the end, we need to rewind by 1,
         * because mapped_file+file_size-1 is the last byte 
         */ 
        if ( (pos > 1 && pos-1 < 0.0001) || (pos<1 && 1-pos< 0.0001))
            reader->base->offset --;
    }
}


guint64 get_num_of_cache_lines(reader_t *const reader){
    if (reader->base->total_num !=0 && reader->base->total_num != -1)
        return reader->base->total_num;
    
    guint64 old_offset = reader->base->offset;
    reader->base->offset = 0;
    guint64 num_of_lines = 0;
    // reset_reader(reader);
    
    switch (reader->base->type) {
        case CSV:
            /* same as plain text, except when has_header, it needs to reduce by 1  */
        case PLAIN:
            ;
            char *line_end = NULL;
            long line_len;
            while (!find_line_ending(reader, &line_end, &line_len)){
                reader->base->offset = (void*)line_end - reader->base->mapped_file;
                num_of_lines ++;
            }
            num_of_lines++;
            if (reader->base->type == 'c')
                if (((csv_params_t*)(reader->reader_params))->has_header)
                    num_of_lines --;
            break;
        case BINARY:
        case VSCSI:
            return reader->base->total_num;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
    reader->base->total_num = num_of_lines;
    reader->base->offset = old_offset;
    return num_of_lines;
}


reader_t* clone_reader(reader_t *const reader_in){
    /* this function clone the given reader to give an exactly same reader */
    
    reader_t *const reader = setup_reader(reader_in->base->file_loc,
                                          reader_in->base->type,
                                          reader_in->base->data_type,
                                          reader_in->base->block_unit_size,
                                          reader_in->base->disk_sector_size,
                                          reader_in->base->init_params); 
    memcpy(reader->sdata, reader_in->sdata, sizeof(reader_data_share_t));
    
    // this is not ideal, but we don't want to multiple mapped files
    munmap (reader->base->mapped_file, reader->base->file_size);
    reader->base->mapped_file = reader_in->base->mapped_file;
    reader->base->offset = reader_in->base->offset;
    reader->base->total_num = reader_in->base->total_num; 

    if (reader->base->type == CSV){
        csv_params_t* params = reader->reader_params;
        csv_params_t* params_in = reader_in->reader_params;
        
        fseek(reader->base->file, ftell(reader_in->base->file), SEEK_SET);
        memcpy(params->csv_parser, params_in->csv_parser, sizeof(struct csv_parser));
    }
    return reader;
}




int close_reader(reader_t *const reader){
    /* close the file in the reader or unmmap the memory in the file
     then free the memory of reader object
     Return value: Upon successful completion 0 is returned.
     Otherwise, EOF is returned and the global variable errno is set to
     indicate the error.  In either case no further
     access to the stream is possible.*/
    
    switch (reader->base->type) {
        case CSV:
            ;
            csv_params_t *params = reader->reader_params;
            fclose(reader->base->file);
            csv_free(params->csv_parser);
            g_free(params->csv_parser);
            break;
        case PLAIN:
            fclose(reader->base->file);
            break;
        case BINARY:
        case VSCSI:
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
    }
    
    munmap (reader->base->mapped_file, reader->base->file_size);
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


int close_reader_unique(reader_t *const reader){
    /* close the file in the reader or unmmap the memory in the file
     then free the memory of reader object
     Return value: Upon successful completion 0 is returned.
     Otherwise, EOF is returned and the global variable errno is set to
     indicate the error.  In either case no further
     access to the stream is possible.*/
    
    switch (reader->base->type) {
        case CSV:
            ;
            csv_params_t *params = reader->reader_params;
            fclose(reader->base->file);
            csv_free(params->csv_parser);
            g_free(params->csv_parser);
            break;
        case PLAIN:
            fclose(reader->base->file);
            break;
        case BINARY:
        case VSCSI:
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



// not supported 

int read_one_request_all_info(reader_t *const reader, void* storage){
    /* read one cache line from reader,
     and store it in the pre-allocated memory pointed at storage.
     return 1 when finished or error, otherwise, return 0.
     */
    switch (reader->base->type) {
        case CSV:
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case PLAIN:
            if (fscanf(reader->base->file, "%s", (char*)storage) == EOF)
                return 1;
            else {
                if (strlen((char*)storage)==2 && ((char*)storage)[1] == '\0'
                    && (((char*)storage)[0] == FILE_LF || ((char*)storage)[0]==FILE_CR) )
                    return read_one_request_all_info(reader, storage);
                return 0;
            }
            break;
        case BINARY:
        case VSCSI:
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

void set_no_eof(reader_t *const reader){
    // remove eof flag for reader 
    switch (reader->base->type) {
        case CSV:
            csv_set_no_eof(reader);
            break;
        case PLAIN:
        case BINARY:
        case VSCSI:
            break;
        default:
            ERROR("cannot recognize reader type, given reader type: %c\n",
                  reader->base->type);
            exit(1);
            break;
    }
}



guint64 read_one_timestamp(reader_t *const reader){
    return 0;
}


void read_one_op(reader_t *const reader, void* op){
    ;
}

guint64 read_one_request_size(reader_t *const reader){
    return 0;
}

cache_line_t* new_cacheline(){
    cache_line* cp = g_new0(cache_line, 1);
    cp->op = -1;
    cp->size = 0;
    cp->block_unit_size = 0;
    cp->disk_sector_size = 0; 
    cp->valid = TRUE;
    cp->item_p = (gpointer)cp->item;
    cp->ts = 0;
    cp->real_time = -1; 
    
    return cp;
}


cache_line_t *copy_cache_line(cache_line_t *cp){
    cache_line_t* cp_new = g_new0(cache_line_t, 1);
    memcpy(cp_new, cp, sizeof(cache_line_t));    
    cp_new->item_p = (gpointer)cp_new->item;
    return cp_new;
}


void destroy_cacheline(cache_line_t* cp){
    if (cp->content)
        g_free(cp->content); 
    g_free(cp);
}


#ifdef __cplusplus
}
#endif
