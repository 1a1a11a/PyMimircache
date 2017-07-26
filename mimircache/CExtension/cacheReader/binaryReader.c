//
//  binaryReader.c
//  mimircache
//
//  Created by Juncheng on 2/28/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "binaryReader.h"

#ifdef __cplusplus
extern "C"
{
#endif



int binaryReader_setup(const char *const filename,
                       reader_t *const reader,
                       const binary_init_params_t *const init_params){

    /* passed in init_params needs to be saved within reader, 
     * to faciliate clone and free */
    reader->base->init_params = g_new(binary_init_params_t, 1);
    memcpy(reader->base->init_params, init_params, sizeof(binary_init_params_t));
    
	reader->base->type = 'b';
    reader->base->record_size = 0;

    reader->reader_params = g_new0(binary_params_t, 1);
    binary_params_t* params = reader->reader_params;
    strcpy(params->fmt, init_params->fmt);
    
    
    /* begin parsing input params and fmt */
    const char *cp = init_params->fmt;
    // ignore the first few characters related to endien
    while (! ((*cp>='0' && *cp<='9') || (*cp>='a' && *cp<='z') || (*cp>='A' && *cp<='Z'))){
        cp ++;
    }
    
    int count = 0, last_count_sum = 0;
    int count_sum = 0;
    int size = 0;
    
    while (*cp){
        count = 0;
        while (*cp>='0' && *cp<='9'){
            count *= 10;
            count += *cp - '0'; 
            cp ++;
        }
        if (count == 0)
            count = 1;      // does not have a number prepend format character
        last_count_sum = count_sum;
        count_sum += count;
        
        switch (*cp) {
            case 'b':
            case 'B':
            case 'c':
            case '?':
                size = 1;
                break;
                
            case 'h':
            case 'H':
                size = 2;
                break;
                
            case 'i':
            case 'I':
            case 'l':
            case 'L':
                size = 4;
                break;
                
            case 'q':
            case 'Q':
                size = 8;
                break;
                
            case 'f':
                size = 4;
                break;
                
            case 'd':
                size = 8;
                break;
                
            case 's':
                size = 1;
                // for string, it should be only one
                count_sum = count_sum - count + 1;
                break;
                
                
            default:
                ERROR("can NOT recognize given format character: %c\n", *cp);
                break;
        }
        
        if (init_params->label_pos != 0 &&
            params->label_len == 0 &&
            init_params->label_pos <= count_sum){
            params->label_pos = (gint)reader->base->record_size +
                    size * (init_params->label_pos - last_count_sum - 1);
            params->label_len = *cp=='s'?count:1;
            params->label_type = *cp;
            // important! update data type here 
            reader->base->data_type = *cp=='s'?'c':'l';
        }

        if (init_params->op_pos != 0 &&
            params->op_len == 0 &&
            init_params->op_pos <= count_sum){
            params->op_pos = (gint)reader->base->record_size +
            size * (init_params->op_pos - last_count_sum - 1);
            params->op_len = *cp=='s'?count:1;
            params->op_type = *cp;
        }

        if (init_params->real_time_pos != 0 &&
            params->real_time_len == 0 &&
            init_params->real_time_pos <= count_sum){
            params->real_time_pos = (gint)reader->base->record_size +
            size * (init_params->real_time_pos - last_count_sum - 1);
            params->real_time_len = *cp=='s'?count:1;
            params->real_time_type = *cp;
        }

        if (init_params->size_pos != 0 &&
            params->size_len == 0 &&
            init_params->size_pos <= count_sum){
            params->size_pos = (gint)reader->base->record_size +
            size * (init_params->size_pos - last_count_sum - 1);
            params->size_len = size;
            params->size_type = *cp;
        }
        
        reader->base->record_size += count * size;
        cp ++;
    }

    // ASSERTION
    if (init_params->label_pos == -1){
        ERROR("label position cannot be -1\n");
        exit(1); 
    }
    
    if (reader->base->file_size % reader->base->record_size != 0){
        WARNING("trace file size %lu is not multiple of record size %lu, mod %lu\n",
                (unsigned long)reader->base->file_size,
                (unsigned long)reader->base->record_size,
                (unsigned long)reader->base->file_size % reader->base->record_size);
    }
    
    reader->base->total_num = reader->base->file_size/(reader->base->record_size);
    params->num_of_fields = count_sum;
    
    DEBUG_MSG("record size %zu, label pos %d, label len %d, label type %c, "
          "real time pos %d, real time len %d, real time type %c\n",
          reader->base->record_size, params->label_pos, params->label_len, params->label_type,
          params->real_time_pos, params->real_time_len, params->real_time_type); 
    
	return 0;
}




#ifdef __cplusplus
}
#endif
