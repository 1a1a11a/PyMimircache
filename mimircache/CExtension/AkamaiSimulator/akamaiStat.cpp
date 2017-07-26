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
#include "akamaiStat.hpp"



namespace akamaiSimulator {


    
    double akamaiStat::get_avg_latency_L1(){
        unsigned long long hit_layer1 = this->hit[0].load();
        unsigned long long req_layer1 = this->req[0].load();
        double hit_ratio = hit_layer1/(double) req_layer1;
        return this->L1_latency * hit_ratio;
    }
    
    double akamaiStat::get_avg_latency_L2(){
        unsigned long long hit_layer2 = this->hit[1].load();
        unsigned long long req_layer2 = this->req[1].load();
        double hit_ratio = hit_layer2/(double) req_layer2;
        return this->L2_latency * hit_ratio;
    }
    
    double akamaiStat::get_avg_latency(){
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
        }
        latency += this->origin_latency * miss_ratio;
        
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
