//
//  binaryReader.c
//  mimircache
//
//  Created by Juncheng on 2/28/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "binaryReader.h"


int binaryReader_setup(char *filename, reader_t* reader, binary_init_params_t* init_params)
{
	struct stat st;	
	int f = -1;
	void *mapped_file = NULL;

    
	if ( (f = open (filename, O_RDONLY)) < 0){
		ERROR ("Unable to open '%s'\n", filename);
		return -1;
	}

	if ( (fstat (f, &st)) < 0){
		close (f);
		ERROR("Unable to fstat '%s'\n", filename);
		return -1;
	}

    if ( (mapped_file = (mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, f, 0))) == MAP_FAILED){
		close (f);
		ERROR("Unable to allocate %llu bytes of memory\n", (unsigned long long) st.st_size);
		return -1;
	}

    /* passed in init_params needs to be saved within reader, to faciliate clone and free */
    reader->base->init_params = g_new(binary_init_params_t, 1);
    memcpy(reader->base->init_params, init_params, sizeof(binary_init_params_t));
    

	reader->base->mapped_file = mapped_file;
	reader->base->offset = 0;
	reader->base->type = 'b';
    reader->base->record_size = 0;

    reader->reader_params = g_new0(binary_params_t, 1);
    binary_params_t* params = reader->reader_params;
    strcpy(params->fmt, init_params->fmt);
    
    
    /* begin parsing input params and fmt */
    char *cp = init_params->fmt;
    // ignore the first few characters related to endien
    while (! ((*cp>='0' && *cp<='9') || (*cp>='a' && *cp<='z') || (*cp>='A' && *cp<='Z'))){
        cp ++;
    }
    
    int count = 0;
    int size = 0;
    
    while (*cp){
        count = 0;
        while (*cp>='0' && *cp<='9'){
            count *= 10;
            count += *cp;
            cp ++;
        }
        if (count == 0)
            count = 1;      // does not have a number prepend format character

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
                break;
                
                
            default:
                ERROR("DO NOT recognize given format character: %c\n", *cp);
                break;
        }
        
        
        
    }


			reader->base->total_num = st.st_size/(reader->base->record_size);
		
	close(f);
	return 0;
}



