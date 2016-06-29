#ifndef _TRACE_FORMAT_
#define _TRACE_FORMAT_

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <inttypes.h>
#include "reader.h"


#define MAX_TEST 2

typedef struct {
  uint64_t ts;
  uint32_t len;
  uint64_t lbn;
  uint16_t cmd;
} trace_item_t;

typedef struct {
  uint32_t sn;
  uint32_t len;
  uint32_t nSG;
  uint16_t cmd;
  uint16_t ver;
  uint64_t lbn;
  uint64_t ts;
} trace_v1_record_t;

typedef struct {
  uint16_t cmd;
  uint16_t ver;
  uint32_t sn;
  uint32_t len;
  uint32_t nSG;
  uint64_t lbn;
  uint64_t ts;
  uint64_t rt;
} trace_v2_record_t;

typedef enum {
  VSCSI1 = 1,
  VSCSI2 = 2,
  UNKNOWN
} vscsi_version_t;

static inline vscsi_version_t to_vscsi_version(uint16_t ver)
{
  return((vscsi_version_t)(ver >> 8));
}

static inline void v1_extract_trace_item(void *record, trace_item_t *trace_item)
{
  trace_v1_record_t *rec = record;
  trace_item->ts = rec->ts;
  trace_item->cmd = rec->cmd;
  trace_item->len = rec->len;
  trace_item->lbn = rec->lbn;
}

static inline void v2_extract_trace_item(void *record, trace_item_t *trace_item)
{
  trace_v2_record_t *rec = record;
  trace_item->ts = rec->ts;
  trace_item->cmd = rec->cmd;
  trace_item->len = rec->len;
  trace_item->lbn = rec->lbn;
}

static inline size_t record_size(vscsi_version_t version)
{
  if (version == VSCSI1)
    return(sizeof(trace_v1_record_t));
  else if (version == VSCSI2)
    return(sizeof(trace_v2_record_t));
  else
    return(-1);
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


static inline int vscsi_read_ver1(READER* reader, cache_line* c){
    int i;
    for (i=0; i<1; i++){
        trace_v1_record_t *record = (trace_v1_record_t *)(reader->p + reader->offset);
        c->real_time = record->ts;
        c->size = record->len;
        c->op = record->cmd;
        *((guint64*)(c->item_p)) = record->lbn;
       (reader->offset) += reader->record_size;
    }
    return i;
}


static inline int vscsi_read_ver2(READER* reader, cache_line* c){
    int i;
    for (i=0; i<1; i++){
        trace_v2_record_t *record = (trace_v2_record_t *)(reader->p + reader->offset);
        c->real_time = record->ts;
        c->size = record->len;
        c->op = record->cmd;
        *((guint64*)(c->item_p)) = record->lbn;
        (reader->offset) += reader->record_size;
    }
    return i;
}

static inline int vscsi_read(READER* reader, cache_line* c){
  if (reader->offset >= reader->total_num * reader->record_size){
    c->valid = FALSE;
    return 0;
  }
  
  int (*fptr)(READER*, cache_line*) = NULL;
  switch (reader->ver)
  {
    case VSCSI1:
      fptr = vscsi_read_ver1;
      break;

    case VSCSI2:  
      fptr = vscsi_read_ver2;
      break;
  }
  return fptr(reader, c);
}


int vscsi_setup(char *filename, READER* reader);
vscsi_version_t test_vscsi_version(void *trace);


#endif
