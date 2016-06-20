//
//  utils.c
//  mimircache
//
//  Created by Juncheng on 6/2/16.
//  Copyright © 2016 Juncheng. All rights reserved.
//

#include "utils.h"
#include <math.h>
#include "const.h" 



double get_log_base(guint64 max, guint64 expect_result){
    
    double base = 10;
    double result, prev_result = expect_result;
    while (1){
        result = log((double)max)/log(base);
        if (result>expect_result && prev_result<expect_result)
            return base;
        prev_result = result;
        base = (base - 1)/2 + 1;
    }
}




//int main(int argc, char* argv[]){
//    printf("%lf\n", get_log_base(1000, 10));
//    return 0;
//}