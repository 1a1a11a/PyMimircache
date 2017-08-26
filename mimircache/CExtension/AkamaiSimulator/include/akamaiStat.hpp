//
//  akamaiStat.hpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/23/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#ifndef AKAMAI_STAT_HPP
#define AKAMAI_STAT_HPP





#ifdef __cplusplus
extern "C"
{
#endif
    
#include <stdio.h>
#include <glib.h>
#include "logging.h" 
    
#ifdef __cplusplus
}
#endif


#include <atomic> 
#include "constAkamaiSimulator.hpp"





namespace akamaiSimulator {
    

    class akamaiStat{
        
        
    public:
        
        /** should we move all the stat to here
         *  instead of placing them inside serverThread and layerThread */
        
        /* number of requests */

        
        std::atomic_ullong req[NUM_CACHE_LAYERS];
        std::atomic_ullong hit[NUM_CACHE_LAYERS];
        
        
        unsigned long L1_latency;
        unsigned long L2_latency;
        unsigned long origin_latency;
        unsigned long layer_latency[NUM_CACHE_LAYERS];
        
        
        akamaiStat(const unsigned long L1_latency,
                   const unsigned long L2_latency,
                   const unsigned long origin_latency); 
        
        // akamaiStat(unsigned long num_servers);

        double get_hr_L1();
        double get_hr_L2();
        double get_avg_latency();
        
        double get_traffic_to_origin();
        double get_traffic_between_first_second_layer();
        
        
    };
}






#endif  /* AKAMAI_STAT_HPP */


