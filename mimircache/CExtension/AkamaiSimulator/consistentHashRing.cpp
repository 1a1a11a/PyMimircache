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

    consistantHashRing::consistantHashRing(){
        this->c_hash_ring=NULL;
        this->find_avail_identifier();
    }
    
    
    void consistantHashRing::find_avail_identifier(){
        this->identifier = 1;
        snprintf(this->identifier_path, 120, "/tmp/%d", this->identifier);
        while (fopen(this->identifier_path, "r") != NULL){
            this->identifier ++;
            snprintf(this->identifier_path, 120, "/tmp/%d", this->identifier);
        }
    }
    
    void consistantHashRing::build_ring(int num_servers, const double* const weight) {
//        ::ketama_build_hashring(&(this->c_hash_ring), num_servers, weight,
//                                CONSISTENT_HASHRING_KEY_IDENTIFIER);
        
        /** we cannot use a single const key identifier as we may run multiple instances
         *  in order to solve this problem, we generate a random number,
         *  check whether it is in /tmp/ or not, if yes, then generate another,
         *  keep doing it, until find an available number. */
        
        /* generating buffer for time format */
        char buffer[30];
        struct timeval tv;
        time_t curtime;
        
        gettimeofday(&tv, NULL);
        curtime = tv.tv_sec;
        strftime(buffer, 30, "%m-%d-%Y %T", localtime(&curtime));

        
        if (this->identifier_path[0] == 0){
            this->find_avail_identifier();
        }
        
        FILE *f = fopen(this->identifier_path, "w");
        fprintf(f, "experiment begin on %s\n", buffer);
        fclose(f);
        
        info("consistent hash ring initialized using identifier %d\n", this->identifier);
        
        ::ketama_build_hashring(&(this->c_hash_ring), num_servers, weight, this->identifier);
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

    int consistantHashRing::get_server_index(char* content){
        return ::ketama_get_server_index( this->c_hash_ring, (const char*)content );
    }
    
    
    
    
    consistantHashRing::~consistantHashRing(){
        if (this->c_hash_ring != NULL)
            ketama_smoke(this->c_hash_ring);
        
        /* remove /tmp/identifier */
        remove(this->identifier_path);
    }
}
