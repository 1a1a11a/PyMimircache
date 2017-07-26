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


#define CONSISTENT_HASHRING_KEY_IDENTIFIER 2



/** currently consistent hash ring does not support change after initialization */



namespace akamaiSimulator {
    
    enum hashType{
        MD5,
        SHA1 
    };
    
    
    
    class consistantHashRing{
        ketama_continuum c_hash_ring;

        
    public:
        consistantHashRing() {this->c_hash_ring=NULL;}
        consistantHashRing(int num_servers, const double* const weight=NULL);
        void build_ring(int num_servers, const double* const weight);
        int get_server_index(cache_line_t *cp);
        
        
        ~consistantHashRing();
    };
    
    
    
}



#endif /* consistentHashRing_hpp */
