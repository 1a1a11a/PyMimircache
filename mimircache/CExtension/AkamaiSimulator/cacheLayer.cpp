//
//  cacheLayer.cpp
//  mimircache
//
//  Created by Juncheng Yang on 7/13/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "cacheLayer.hpp"




namespace akamaiSimulator {
    
    
    cacheLayerStat::cacheLayerStat(const unsigned long num_server,
                                   const unsigned long layer_id) {
        this->num_server = num_server;
        this->layer_id = layer_id;
        this->num_req_from_server = new unsigned long[num_server];
        this->num_req_to_server = new unsigned long[num_server];
        this->per_server_hit_count = new unsigned long[num_server];
        this->weight = new double[num_server];
        this->layer_hit_count = 0;
        this->layer_req_count = 0;
        
        for (unsigned int i=0; i<num_server; i++){
            this->num_req_from_server[i] = 0;
            this->num_req_to_server[i] = 0;
            this->per_server_hit_count[i] = 0;
            this->weight[i] = 0;
        }
    }
    
    cacheLayerStat::cacheLayerStat(const cacheLayerStat& stat){
        *this = stat;
    }
    
    cacheLayerStat& cacheLayerStat::operator=(const akamaiSimulator::cacheLayerStat &stat){
        if (this == &stat)
            return *this;
        
        this->num_server = stat.num_server;
        this->layer_id = stat.layer_id;
        this->layer_hit_count = stat.layer_hit_count;
        this->layer_req_count = stat.layer_req_count;
        
        this->num_req_from_server = new unsigned long[this->num_server];
        this->num_req_to_server = new unsigned long[this->num_server];
        this->per_server_hit_count = new unsigned long[this->num_server];
        this->weight = new double[this->num_server]; 
        
        for (unsigned int i=0; i<this->num_server; i++){
            this->num_req_from_server[i] = stat.num_req_from_server[i];
            this->num_req_to_server[i] = stat.num_req_to_server[i];
            this->per_server_hit_count[i] = stat.per_server_hit_count[i];
            this->weight[i] = stat.weight[i];
        }
        
        return *this;
    }
    
    void cacheLayerStat::set_weight(double *weight) {
        memcpy(this->weight, weight, sizeof(double)*this->num_server);
    }
    
    
    cacheLayerStat::~cacheLayerStat() {
        delete [] this->num_req_from_server;
        delete [] this->num_req_to_server;
        delete [] this->per_server_hit_count;
        delete [] this->weight;
    }
    
    
    
    
    
    
    
    
    
    
    cacheLayer::cacheLayer(std::vector<cacheServer*> cache_servers,
                           akamaiStat *akamai_stat,
                           int layer_id, enum hashType hash_type){
        
        this->akamai_stat = akamai_stat;
        this->layer_id = layer_id;
        this->cache_servers = cache_servers;
        this->next_layer = NULL;
        this->weight = new double[cache_servers.size()];
        this->cal_weight();
        
        /* initialize cache layer stat struct */
        this->layer_stat = new cacheLayerStat(cache_servers.size(), this->layer_id);
        this->layer_stat->set_weight(weight);
        
        
        if (hash_type == MD5){
            this->ring.build_ring((int)cache_servers.size(), this->weight);
        }
        else{
            error_msg("hash type %d not supported\n", hash_type);
            abort();
        }
        
    }
    
    cacheLayer::cacheLayer(cacheServer** cache_servers,
                           const unsigned long num_servers,
                           akamaiStat *akamai_stat,
                           int layer_id, enum hashType hash_type){
        
        this->akamai_stat = akamai_stat;
        this->layer_id = layer_id;
        this->cache_servers = std::vector<cacheServer*>();
        for (unsigned long i =0; i<num_servers; i++)
            this->cache_servers.push_back(cache_servers[i]);
        
        this->next_layer = NULL;
        this->weight = new double[num_servers];
        this->cal_weight();
        
        /* initialize cache layer stat struct */
        this->layer_stat = new cacheLayerStat(num_servers, this->layer_id);
        this->layer_stat->set_weight(this->weight);
        
        
        if (hash_type == MD5){
            this->ring.build_ring((int)num_servers, this->weight);
        }
        else{
            error_msg("hash type %d not supported\n", hash_type);
            abort();
        }
        
    }
    
    
    void cacheLayer::cal_weight(){
        static bool first_flag = true;
        if (first_flag){
            first_flag = false;
        }
        else{
            warning("I don't understand why we need to adjust the "
                    "weight of cache server in simulation\n");
            abort();
        }
        
        
        std::vector<cacheServer*>::const_iterator it = this->cache_servers.cbegin();
        while (it < this->cache_servers.cend()){
            /** use layer_id - 1 here because layer_id begins from 1,
             *  but array begins from 0 */
            this->weight[it - this->cache_servers.cbegin()] =
                ((*it)->get_layer_size()[this->layer_id-1]);
            it++;
        }
    }
    
    void cacheLayer::rebalance(){
        error_msg("this function is not implemented yet\n");
        abort();
        
        
    }
    
    
    /** modify this part if a request needs to be added to more than one server in the layer 
     *  This is NON-BLOCKING for now, all requests directly go to cache server and 
     *  being added to corresponding cache 
     */
    void cacheLayer::add_request(const unsigned long cache_server_id,
                                     cache_line_t * const cp){
        
        /** first get the index of the cacheServer which 
         *  this request will be sent to using consistent hashing */
        
        unsigned long server_ind;
        gboolean cache_hit = FALSE;
        gboolean cache_hit_current_server;
        char temp[CACHE_LINE_LABEL_SIZE];
        
        memcpy(temp, cp->item, CACHE_LINE_LABEL_SIZE);
        
        for (unsigned long i=0; i<L2_COPY; i++){
            size_t len = strlen(temp);
            len = len<=CACHE_LINE_LABEL_SIZE-2 ? len: CACHE_LINE_LABEL_SIZE-2;
            temp[len] = i+1;
            temp[len+1] = 0;
            server_ind = this->ring.get_server_index(temp);
            
            cache_hit_current_server = this->cache_servers.at(server_ind)->add_request(cp, this->layer_id);
            this->layer_stat->num_req_to_server[server_ind] ++;
            
            if (cache_hit_current_server){
                cache_hit = TRUE;
                this->layer_stat->per_server_hit_count[server_ind] ++;
            }
        }
        
        
        /* update layer stat */
        this->layer_stat->num_req_from_server[cache_server_id] ++;
        if (cache_hit){
            this->layer_stat->layer_hit_count ++;
        }
        
        this->layer_stat->layer_req_count ++; 
//        this->layer_stat->layer_req_count += L2_COPY;

        /* end of layer stat update */
    }
    
    
    void cacheLayer::add_server(cacheServer* cache_server){
        error_msg("this function is not implemented yet\n");
        abort();
    }
    
    void cacheLayer::set_next_layer(akamaiSimulator::cacheLayer *next_layer){
        this->next_layer = next_layer; 
    }
    
    
    cacheServer& cacheLayer::get_server(const int index){
        return *(this->cache_servers.at(index));
    }
    
    cacheLayer* cacheLayer::get_next_layer(){
        return this->next_layer;
    }
    
    int cacheLayer::get_server_index(cache_line_t* const cp){
        return this->ring.get_server_index(cp);
    }
    
    size_t cacheLayer::get_num_server(){
        return this->cache_servers.size();
    }
    
    int cacheLayer::get_layer_id(){
        return this->layer_id;
    }
    
    
    cacheLayerStat* cacheLayer::get_layer_stat(){
        cacheLayerStat *stat = new cacheLayerStat(*(this->layer_stat));
        return stat;
    }
    
    
    cacheLayer::~cacheLayer(){
        delete [] this->weight;
        delete this->layer_stat;
    }
}
