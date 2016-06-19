//
//  splay.h
//  LRUAnalyzer
//
//  Created by Juncheng on 5/25/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#ifndef splay_h
#define splay_h


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#define node_value(x) (((x)==NULL) ? 0 : ((x)->value))
#define key_cmp(i,j) ((i)-(j))

typedef long long cache_line_type;



typedef struct sTree{
    struct sTree * left, * right;
    cache_line_type key;
    long value;
}sTree;





void static inline free_node(sTree* t){
    free(t);
}
void static inline assign_key(sTree* t, cache_line_type k){
    t->key=k;
}





sTree * splay (cache_line_type i, sTree *t);
sTree * insert(cache_line_type i, sTree * t);
sTree * delete(cache_line_type i, sTree *t);
sTree *find_node(cache_line_type r, sTree *t);
void check_sTree(sTree* t);
void print_sTree(sTree * t, int d);
void free_sTree(sTree* t);


#endif /* splay_h */
