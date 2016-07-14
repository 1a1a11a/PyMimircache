//
//  glib_related.h
//  LRUAnalyzer
//
//  Created by Juncheng on 5/26/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef glib_related_h
#define glib_related_h

#include <stdio.h>
#include <stdlib.h>
#include <glib.h> 
#include "pqueue.h"





void simple_key_value_destroyer(gpointer data);
void simple_g_key_value_destroyer(gpointer data);
void g_slist_destroyer(gpointer data);
void gqueue_destroyer(gpointer data);
void pqueue_node_destroyer(gpointer data);


#endif /* glib_related_h */
