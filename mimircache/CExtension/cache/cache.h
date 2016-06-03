//
//  cache.h
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef cache_h
#define cache_h

#include <stdio.h>
#include <glib.h>
#include "reader.h"

struct cache{
    char name[32];
    long size;
    gblooean (*add_element)(cache_line* cp);
    gblooean (*check_element)(cache_line* cp);
    
};






#endif /* cache_h */
