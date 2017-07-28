//
//  akamaiStat.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/23/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//




#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <sstream>
#include "akamaiStat.hpp"



namespace akamaiSimulator {

    
    akamaiStat::akamaiStat(const unsigned long L1_latency,
                           const unsigned long L2_latency,
                           const unsigned long origin_latency){
        
        this->L1_latency = L1_latency;
        this->L2_latency = L2_latency;
        this->origin_latency = origin_latency;
        
        for (unsigned long i=0; i<NUM_CACHE_LAYERS; i++){
            this->req[i].store(0);
            this->hit[i].store(0); 
        }
        this->layer_latency[0] = L1_latency;
        this->layer_latency[1] = L2_latency;
    }

    
    double akamaiStat::get_hr_L1(){
        unsigned long long hit_layer1 = this->hit[0].load();
        unsigned long long req_layer1 = this->req[0].load();
        double hit_ratio = hit_layer1/(double) req_layer1;
        return hit_ratio;
    }
    
    double akamaiStat::get_hr_L2(){
        unsigned long long hit_layer2 = this->hit[1].load();
        unsigned long long req_layer2 = this->req[1].load();
        double hit_ratio = hit_layer2/(double) req_layer2;
        return hit_ratio;
    }
    
    double akamaiStat::get_avg_latency(){
        std::stringstream ss;
        unsigned long long hit_layer;
        unsigned long long req_layer;
        double hit_ratio, miss_ratio = 1;   /* miss_ratio is the miss_ratio of overall miss_ratio */
        double latency = 0;
        for (int i=0; i<NUM_CACHE_LAYERS; i++){
            hit_layer = this->hit[i].load();
            req_layer = this->req[i].load();
            hit_ratio = miss_ratio * (hit_layer/(double) req_layer);
            latency += this->layer_latency[i] * hit_ratio;
            miss_ratio -= hit_ratio;
            ss<<hit_ratio<<":"<<miss_ratio<<", ";
        }
        latency += this->origin_latency * miss_ratio;
        ss<<miss_ratio;
        info("hit/miss ratio %s\n", ss.str().c_str());
        return latency;
        
    }
    
    double akamaiStat::get_traffic_to_origin(){
        unsigned long long hit_layer;
        unsigned long long req_layer;
        double hit_ratio, miss_ratio = 1;   /* miss_ratio is the miss_ratio of overall miss_ratio */
        for (int i=0; i<NUM_CACHE_LAYERS; i++){
            hit_layer = this->hit[i].load();
            req_layer = this->req[i].load();
            hit_ratio = miss_ratio * (hit_layer/(double) req_layer);
            miss_ratio -= hit_ratio;
        }
        return miss_ratio;
    }
    
    
    double akamaiStat::get_traffic_between_first_second_layer(){
        unsigned long long hit_layer1 = this->hit[0].load();
        unsigned long long req_layer1 = this->req[0].load();
        double hit_ratio = hit_layer1/(double) req_layer1;
        return 1 - hit_ratio; 
    }
}
