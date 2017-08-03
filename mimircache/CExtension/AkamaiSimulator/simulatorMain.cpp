//
//  simulatorMain.cpp
//  mimircache
//
//  Created by Juncheng Yang on 7/11/17.
//  Copyright © 2017 Juncheng. All rights reserved.
//

#include "simulatorMain.hpp"


#include <iostream> 
#include <fstream>
#include <string>


#define AKAMAI_DATA_TYPE 0



std::vector<std::string> get_traces_from_config(std::string config_loc){
    
    std::ifstream ifs(config_loc, std::ios::in);
    std::vector<std::string> traces;
    std::string line;
    while (getline(ifs, line)){
        traces.push_back(line);
    }
    
    ifs.close();
    return traces;
}


void parse_cmd_arg(int argc, char* argv[], simulator_arg_t* sargs){
    

    (*sargs).cache_size = 0;
    if (argc < 7){
        std::cerr << "usage: " << argv[0] << "-c config -s size -t traceType\n";
        exit(0);
    }
    (*sargs).log_folder = "log";
    
    int opt;
    extern char *optarg;
    // extern int optind, optopt;
    
    // Retrieve the options:
    while ( (opt = getopt(argc, argv, "b:c:o:s:t:")) != -1 ) {  // for each option...
        switch ( opt ) {
            case 'b':
                (*sargs).boundary = atof(optarg);
                break;
            case 'c':
                (*sargs).config_loc = std::string(optarg);
                break;
            case 'o':
                printf("get option o %s\n", optarg);
                (*sargs).log_folder = std::string(optarg);
                break; 
            case 's':
                (*sargs).cache_size = (unsigned long) atol(optarg);
                break;
            case 't':
                (*sargs).trace_type = (unsigned long) atoi(optarg);
                break;
        }
    }
    info("trace config: %s, size %lu, boundary %lf, type %lu\n",
         (*sargs).config_loc.c_str(), (*sargs).cache_size,
         (*sargs).boundary, (*sargs).trace_type);
}






int main(int argc, char* argv[]){
    
    
    
    simulator_arg_t arg;
    parse_cmd_arg(argc, argv, &arg);
    
    
    
    if (arg.cache_size == 0 || arg.config_loc.size() == 0){
        std::cerr << "please provide both size and config file loc" << std::endl;
        return 0; 
    }
    
//    std::vector<std::string> traces = {"/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample2/a",
//        "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample2/b"}; //,
    //    std::vector<std::string> traces = {"/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample/60.14.244.130",
    //        "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample/60.14.244.144"}; //,
    //        "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample/60.14.244.159",
    //        "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample/60.14.244.163",
    //        "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/sample/60.14.244.176"};
//    arg.cache_size = 10000;

    auto traces = get_traces_from_config(arg.config_loc);
    
    double boundaries[NUM_CACHE_LAYERS];
//    std::fill_n(boundaries, NUM_CACHE_LAYERS, (double)1/NUM_CACHE_LAYERS);
    std::fill_n(boundaries, NUM_CACHE_LAYERS, arg.boundary);
    unsigned long cache_sizes[traces.size()];
    std::fill_n(cache_sizes, traces.size(), arg.cache_size);

    akamaiSimulator::akamai_run(traces, boundaries, cache_sizes, arg.trace_type, arg.log_folder, false);
    
    
    return 1;
}
