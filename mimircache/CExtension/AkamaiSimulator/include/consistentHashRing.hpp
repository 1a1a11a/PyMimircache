//
//  consistentHashRing.hpp
//  akamaiSimulator
//
//  Created by Juncheng Yang on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#ifndef consistentHashRing_hpp
#define consistentHashRing_hpp


#ifdef __cplusplus
extern "C"
{
#endif
    
#include <stdio.h>
#include <glib.h>
#include <stdlib.h> 
    
#include "cache.h"
#include "cacheHeader.h"
#include "reader.h"
#include "ketama.h"
#include "logging.h"
    
#ifdef __cplusplus
}
#endif


#include <vector>
#include "constAkamaiSimulator.hpp"





/** currently consistent hash ring does not support change after initialization */



namespace akamaiSimulator {
    
    enum hashType{
        MD5,
        SHA1 
    };
    
    
    
    class consistantHashRing{
        ketama_continuum c_hash_ring;
        int identifier;
        char identifier_path[128]; 
        
    public:
        consistantHashRing();
        consistantHashRing(int num_servers, const double* const weight=NULL);
        void find_avail_identifier();
        void build_ring(int num_servers, const double* const weight);
        int get_server_index(cache_line_t *cp);
        int get_server_index(char *cp);
        
        
        ~consistantHashRing();
    };
    
    
    
}



#endif /* consistentHashRing_hpp */
