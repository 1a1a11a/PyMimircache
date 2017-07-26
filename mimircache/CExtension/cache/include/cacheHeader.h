//
//  cacheHeader.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef CACHEHEADER_H
#define CACHEHEADER_H




#ifdef __cplusplus
extern "C" {
#endif
    
    
    
#include "FIFO.h"
#include "Optimal.h"
#include "LRU_K.h"
#include "LRU.h"
#include "LFU.h"
#include "LFUFast.h"
#include "MRU.h"
#include "Random.h"
#include "ARC.h"
#include "SLRU.h"
    
#ifdef ML
#include "SLRUML.h"
#include "Score.h"
#endif
//#include "LRFU.h"
    
#include "mimir.h"
#include "AMP.h"
#include "PG.h"


       
        
#ifdef __cplusplus
}
#endif


#endif /* cacheHeader_h */
