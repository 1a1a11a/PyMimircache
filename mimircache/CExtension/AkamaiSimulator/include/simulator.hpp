//
//  simulator.cpp
//  akamaiSimulator
//
//  Created by Juncheng on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//


#ifndef simulator_HPP
#define simulator_HPP





#ifdef __cplusplus
extern "C"
{
#endif
    
#include <stdio.h>
#include <glib.h>
#include <string.h> 
    
#include "cache.h"
#include "cacheHeader.h"
#include "reader.h"
#include "csvReader.h"
#include "utils.h" 
    
#ifdef __cplusplus
}
#endif



#include "constAkamaiSimulator.hpp"
#include "cacheLayer.hpp"
#include "cacheServer.hpp"
#include "cacheLayerThread.hpp"
#include "cacheServerThread.hpp"
#include "akamaiStat.hpp"




#define AKAMAI1_CSV_PARAM_INIT (new_csvReader_init_params(5, -1, 1, -1, FALSE, '\t', -1))
#define AKAMAI3_CSV_PARAM_INIT (new_csvReader_init_params(6, -1, 1, -1, FALSE, '\t', -1))
#define AKAMAI0_CSV_PARAM_INIT (new_csvReader_init_params(2, -1, 1, -1, FALSE, '\t', -1))


namespace akamaiSimulator {
    
    
    akamaiStat* akamai_run(std::vector<std::string> traces,
                    double *boundaries,
                    unsigned long* cache_sizes,
                    unsigned long akamai_data_type,
                    const std::string log_folder,
                    bool dynamic_boundary_flag);

}






#endif


