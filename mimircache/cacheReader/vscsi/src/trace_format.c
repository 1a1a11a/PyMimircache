#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <inttypes.h>

#include "trace_format.h"

long get_label_trace_v1(void* data){
   trace_v1_record_t *record = (trace_v1_record_t *)data;
   return record->lbn; 
}

long get_label_trace_v2(void* data){
   trace_v2_record_t *record = (trace_v2_record_t *)data;
   return record->lbn; 
}

int set_return_value_v1(void** mem, int size, long ts[], int len[], long lbn[], int cmd[], int delta){
   int i;
   for (i=0; i<size; i++){
       trace_v1_record_t *record = (trace_v1_record_t *)(*mem);
       ts[i] = record->ts;
       len[i] = record->len;
       cmd[i] = record->cmd;
       lbn[i] = record->lbn;
       *mem += delta; 
   }
   return i;
}


int set_return_value_v2(void** mem, int size, long ts[], int len[], long lbn[], int cmd[], int delta){
   int i;
   for (i=0; i<10000; i++){
       trace_v2_record_t *record = (trace_v2_record_t *)(*mem);
       ts[i] = record->ts;
       len[i] = record->len;
       cmd[i] = record->cmd;
       lbn[i] = record->lbn;
       *mem += delta; 
   }
   return i;
}





void print_trace_v1(FILE *stream, void *data)
{
   trace_v1_record_t *record = (trace_v1_record_t *)data;
   fprintf(stream, "%hu %" PRId64 " %x %u %" PRId64 "\n",
           (unsigned short)(record->ver >> 8), record->ts,
           record->cmd, record->len, record->lbn);
}

void print_trace_v2(FILE *stream, void *data)
{
   trace_v2_record_t *record = (trace_v2_record_t *)data;
   fprintf(stream, "%hu %" PRId64" %x %u %" PRId64 " %" PRId64"\n",
           (unsigned short)(record->ver >> 8), record->ts,
           record->cmd, record->len, record->lbn,
           record->rt);
}

void print_trace_item(FILE *stream, trace_item_t *item)
{
   fprintf(stream, "%"PRId64" %x %u %" PRId64 "\n",
           item->ts, item->cmd, item->len,
           item->lbn);
}

// XXX: length of test_buf must be MAX_TEST
static inline bool test_version(vscsi_version_t *test_buf,
                                vscsi_version_t version)
{
  int i;
  for (i=0; i<MAX_TEST; i++) {
    if (version != test_buf[i])
      return(false);
  }
  return(true);
}

// XXX: only handle two trace types for now.
vscsi_version_t test_vscsi_version(void *trace)
{
  vscsi_version_t test_buf[MAX_TEST] = {};

  int i;
  for (i=0; i<MAX_TEST; i++) {
    test_buf[i] = (vscsi_version_t)((((trace_v2_record_t *)trace)[i]).ver >> 8);
  }

  if (test_version(test_buf, VSCSI2))
    return(VSCSI2);
  else {
    for (i=0; i<MAX_TEST; i++) {
      test_buf[i] = (vscsi_version_t)((((trace_v1_record_t *)trace)[i]).ver >> 8);
    }
    if (test_version(test_buf, VSCSI1))
      return(VSCSI1);
  }

  return(UNKNOWN);
}



