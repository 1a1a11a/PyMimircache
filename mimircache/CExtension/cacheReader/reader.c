//
//  reader.c
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "reader.h"
#include "vscsi_trace_format.h"

// add support for lock in reader to prevent simultaneous access reader



READER* setup_reader(char* file_loc, char file_type){
    /* setup the reader struct for reading trace
     file_type: c: csv, v: vscsi, p: plain text
     Return value: a pointer to READER struct, the returned reader
     needs to be explicitly closed by calling close_reader */
    
    READER* reader = g_new0(READER, 1);
    reader->break_points = NULL;
    reader->last_access = NULL;
    reader->reuse_dist = NULL;
    reader->hit_rate = NULL;
    reader->best_LRU_cache_size = NULL;
    reader->max_reuse_dist = 0;
    reader->total_num = -1;
    
    
    if (strlen(file_loc)>FILE_LOC_STR_SIZE-1){
        printf("file name/path is too long(>%d), please make it short\n", FILE_LOC_STR_SIZE);
        exit(1);
    }
    else{
        strcpy(reader->file_loc, file_loc);
    }

    switch (file_type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':
            reader->type = 'p';
            reader->ts = 0;
            reader->file = fopen(file_loc, "r");
            if (reader->file == 0){
                perror("open trace file failed\n");
                exit(1);
            }
            break;
        case 'v':
            vscsi_setup(file_loc, reader);
            break;
        default:
            printf("cannot recognize trace file type, it can only be c(csv), "
                   "p(plain text), v(vscsi)\n");
            exit(1);
            break;
    }
    return reader;
}

void read_one_element(READER* reader, cache_line* c){
    /* read one cache line from reader, 
     and store it in the pre-allocated cache_line c, current given 
     size for the element(label) is 128 bytes(cache_line_label_size).
     */
    switch (reader->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':

            if (fscanf(reader->file, "%s", c->item) == EOF)
                c->valid = FALSE;
            else {
                if (strlen(c->item)==2 && c->item[0] == '\n' && c->item[1] == '\0')
                    return read_one_element(reader, c);
                c->ts = (reader->ts)++;
            }
            break;
        case 'v':
            vscsi_read(reader, c);
            c->ts = (reader->ts)++;
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), "
                   "v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
}

int go_back_one_line(READER* reader){
    /* go back two cache lines
     return 0 on successful, non-zero otherwise
     */
    switch (reader->type) {
        case 'c':
        case 'p':
            ;
            int r;
            
            fseek(reader->file, -2L, SEEK_CUR);
            while( getc(reader->file) != '\n' )
                if (( r= fseek(reader->file, -2L, SEEK_CUR)) != 0)
                    return r;
            
            return 0;
        case 'v':
            if (reader->offset >= reader->record_size)
                reader->offset -= (reader->record_size);
            else
                return -1;
            return 0;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), "
                   "v(vscsi), but got %c\n", reader->type);
            exit(1);
    }    
}



int go_back_two_lines(READER* reader){
    /* go back two cache lines 
     return 0 on successful, non-zero otherwise 
     */
    switch (reader->type) {
        case 'c':
        case 'p':
            ;
            int r;
            if ( (r=fseek(reader->file, -2L, SEEK_CUR)) != 0){
                return r;
            }
            
            while( getc(reader->file) != '\n' )
                if ( (r= fseek(reader->file, -2L, SEEK_CUR)) != 0){
                    return r;
                }
            fseek(reader->file, -2L, SEEK_CUR);
            while( getc(reader->file) != '\n' )
                if ( (r=fseek(reader->file, -2L, SEEK_CUR)) != 0){
                    fseek(reader->file, -1L, SEEK_CUR);
                    break;
                }

            return 0;
        case 'v':
            if (reader->offset >= (reader->record_size * 2))
                reader->offset -= (reader->record_size)*2;
            else
                return -1;
            return 0;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), "
                   "v(vscsi), but got %c\n", reader->type);
            exit(1);
    }    
}



void read_one_element_above(READER* reader, cache_line* c){
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



long skip_N_elements(READER* reader, long long N){
    /* skip the next following N elements, 
     Return value: the number of elements that are actually skipped, 
     this will differ from N when it reaches the end of file
     */
    int i;
    long long count=0;
    char temp[cache_line_label_size];
    switch (reader->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':
            for (i=0; i<N; i++)
                if (fscanf(reader->file, "%s", temp)!=EOF)
                    count++;
                else
                    break;
            reader->ts += i;
            break;
        case 'v':
            reader->offset = reader->offset + N * reader->record_size;
            reader->ts += N;
            count = N;
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), \
                   v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
    return count;
}

void reset_reader(READER* reader){
    /* rewind the reader back to beginning 
     */
    switch (reader->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':
            fseek(reader->file, 0L, SEEK_SET);
            reader->ts = 0;
            break;
        case 'v':
            reader->offset = 0;
            reader->ts = 0;
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), \
                   v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
}


void reader_set_read_pos(READER* reader, float pos){
    /* jump to given postion, like 1/3, or 1/2 and so on
     * 
     */
    switch (reader->type) {
        case 'c':
        case 'p':
            ;
            struct stat statbuf;
            if (fstat(fileno(reader->file), &statbuf) < 0)
                printf("fstat error");
            long long fsize= (long long)statbuf.st_size;
            fseek(reader->file, (long)(fsize*pos), SEEK_SET);
            char c = getc(reader->file);
            while (c!='\n' && c!=EOF)
                c = getc(reader->file);
            reader->ts = 0;
            break;
            
        case 'v':
            reader->offset = (guint64)reader->record_size * reader->total_num;
            reader->ts = 0;
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), \
                   v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
    
}


guint64 get_num_of_cache_lines(READER* reader){
    
#define BUFFER_SIZE 1024*1024
    guint64 num_of_lines = 0;
    char temp[BUFFER_SIZE+1];       // 5MB buffer
    int fd, i=0;
    long size;
    reset_reader(reader);
    
    switch (reader->type) {
        case 'c':
            // same as plain text
        case 'p':
            fd = fileno(reader->file);
            lseek(fd, 0L, SEEK_SET);
            char last_char = 0; 
            while ((size=read(fd, (void*)temp, BUFFER_SIZE))!=0){
                if (temp[0] == '\n' && last_char != '\n')
                    num_of_lines++;
                for (i=1;i<size;i++)
                    if (temp[i] == '\n' && temp[i-1] != '\n')
                        num_of_lines++;
                last_char = temp[size-1];
            }
            if (last_char!='\n'){
                num_of_lines++;
            }
//            while (fscanf(reader->file, "%s", temp)!=EOF)
//                num_of_lines++;
            break;
        case 'v':
//            printf("vscsi reader has total number of records when initialization, "
//                "so you don't need to call this function\n");
            return reader->total_num;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), \
                   v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
    reader->total_num = num_of_lines;
    reset_reader(reader);
    return num_of_lines;
}


READER* copy_reader(READER* reader_in){
    // duplicate reader
    READER* reader = g_new(READER, 1);
    memcpy(reader, reader_in, sizeof(READER));
    
    if (reader->type == 'v')
        reader->offset = 0;
    else{
        reader->file = fopen(reader->file_loc, "r");
    }
    return reader;
}




int close_reader(READER* reader){
    /* close the file in the reader or unmmap the memory in the file
     then free the memory of reader object
     Return value: Upon successful completion 0 is returned.
     Otherwise, EOF is returned and the global variable errno is set to 
     indicate the error.  In either case no further
     access to the stream is possible.*/

    switch (reader->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            break;
        case 'p':
            fclose(reader->file);
            break;
        case 'v':
            munmap (reader->p, reader->total_num * reader->record_size);
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), \
                   v(vscsi), but got %c\n", reader->type);
            return(1);
    }
    

    if (reader->last_access){
        free(reader->last_access);
    }
    
    if (reader->reuse_dist){
        free(reader->reuse_dist);
    }
    
    if (reader->break_points){
        g_array_free(reader->break_points->array, TRUE);
        free(reader->break_points);
    }
    
    if (reader->hit_rate)
        free(reader->hit_rate);

    if (reader->best_LRU_cache_size)
        g_queue_free(reader->best_LRU_cache_size);

    
    free(reader);
    return 0;
}

int read_one_request_all_info(READER* reader, void* storage){
    /* read one cache line from reader,
     and store it in the pre-allocated memory pointed at storage.
     return 1 when finished or error, otherwise, return 0.
     */
    switch (reader->type) {
        case 'c':
            printf("currently c reader is not supported yet\n");
            exit(1);
            break;
        case 'p':
            if (fscanf(reader->file, "%s", (char*)storage) == EOF)
                return 1;
            else {
                if (strlen((char*)storage)==2 && ((char*)storage)[0] == '\n' && ((char*)storage)[1] == '\0')
                    return read_one_request_all_info(reader, storage);
                return 0;
            }
            break;
        case 'v':
            printf("currently v reader is not supported yet\n");
            exit(1);
            break;
        default:
            printf("cannot recognize reader type, it can only be c(csv), p(plain text), "
                   "v(vscsi), but got %c\n", reader->type);
            exit(1);
            break;
    }
    return 0;
}


guint64 read_one_timestamp(READER* reader){
    return 0;
}


void read_one_op(READER* reader, void* op){
    ;
}

guint64 read_one_request_size(READER* reader){
    return 0;
}

cache_line* new_cacheline(){
    cache_line* cp = g_new(cache_line, 1);
    cp->op = -1;
    cp->size = -1;
    cp->valid = TRUE;
    cp->item_p = (gpointer)cp->item;
    
    return cp;
}

void destroy_cacheline(cache_line* cp){
    g_free(cp);
}