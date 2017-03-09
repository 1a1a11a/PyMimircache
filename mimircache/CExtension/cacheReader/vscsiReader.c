//
//  vscsiReader.c
//  mimircache
//
//  Created by CloudPhysics
//  Modified by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "vscsiReader.h"
#include "reader.h"


int vscsi_setup(char *filename, reader_t* reader){
    struct stat st;
    int f = -1;
    vscsi_params_t* params = g_new0(vscsi_params_t, 1);
    void *mapped_file = NULL;
    
    reader->reader_params = params;
    
    if ( (f = open (filename, O_RDONLY)) < 0){
        ERROR("Unable to open '%s'\n", filename);
        return -1;
    }
    
    if ( (fstat (f, &st)) < 0){
        close (f);
        ERROR("Unable to fstat '%s'\n", filename);
        return -1;
    }
    
    /* Version 2 records are the bigger of the two */
    if ((unsigned long long)st.st_size < (unsigned long long) sizeof(trace_v2_record_t) * MAX_TEST){
        close (f);
        ERROR("File too small, unable to read header.\n");
        return -1;
    }
    
    if ( (mapped_file = (mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, f, 0))) == MAP_FAILED){
        close (f);
        ERROR("Unable to allocate %llu bytes of memory\n", (unsigned long long) st.st_size);
        return -1;
    }
    
    reader->base->mapped_file = mapped_file;
    reader->base->offset = 0;
    reader->base->type = 'v';
    reader->base->data_type = 'l';
    
    
    vscsi_version_t ver = test_vscsi_version (mapped_file);
    switch (ver){
        case VSCSI1:
        case VSCSI2:
            reader->base->record_size = record_size(ver);
            params->vscsi_ver = ver;
            reader->base->total_num = st.st_size/(reader->base->record_size);
            break;
            
        case UNKNOWN:
        default:
            ERROR("Trace format unrecognized.\n");
            return -1;
    }
    close(f);
    return 0;
}


vscsi_version_t test_vscsi_version(void *trace){
    vscsi_version_t test_buf[MAX_TEST] = {};
    
    int i;
    for (i=0; i<MAX_TEST; i++) {
        test_buf[i] = (vscsi_version_t)((((trace_v2_record_t *)trace)[i]).ver >> 8);
    }
    if (test_version(test_buf, VSCSI2))
        return(VSCSI2);
    else {
        for (i=0; i<MAX_TEST; i++) {
            test_buf[i] = (vscsi_version_t)((((trace_v1_record_t *)trace)[i]).ver >> 8);
        }
        
        if (test_version(test_buf, VSCSI1))
            return(VSCSI1);
    }
    return(UNKNOWN);
}


