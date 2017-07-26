//
//  consistentHashRing.cpp
//  akamaiSimulator
//
//  Created by Juncheng Yang on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "consistentHashRing.hpp"


namespace akamaiSimulator {
        
    consistantHashRing::consistantHashRing(int num_servers, const double* const weight) {
        this->build_ring(num_servers, weight);
    }

    
    void consistantHashRing::build_ring(int num_servers, const double* const weight) {
        ::ketama_build_hashring(&(this->c_hash_ring), num_servers, weight,
                                CONSISTENT_HASHRING_KEY_IDENTIFIER);
    }
    
    int consistantHashRing::get_server_index(cache_line_t * const cp) {
        if (this->c_hash_ring == NULL){
            error_msg("consistent hash ring is not initialized\n");
            abort(); 
        }
        if (cp->type == 'c')
            ;
        else if (cp->type == 'l'){
            cp->item[8] = 0;
        }
        else{
            error_msg("unkown cache_line datatype %c\n", cp->type);
            abort();
        }
        return ::ketama_get_server_index( this->c_hash_ring, (const char*)(cp->item_p) );
    }

    
    
    
    
    
    consistantHashRing::~consistantHashRing(){
        if (this->c_hash_ring != NULL)
            ketama_smoke(this->c_hash_ring);
    }
}
