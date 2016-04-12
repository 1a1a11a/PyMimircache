#ifndef _TRACE_FORMAT_
#define _TRACE_FORMAT_

#include <stdint.h>

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

void print_trace_v1(FILE *stream, void *data);
void print_trace_v2(FILE *stream, void *data);
void print_trace_item(FILE *stream, trace_item_t *item);
long get_label_trace_v1(void* data);
long get_label_trace_v2(void* data); 
int set_return_value_v1(void** mem, int size, long ts[], int len[], long lbn[], int cmd[], int delta);
int set_return_value_v2(void** mem, int size, long ts[], int len[], long lbn[], int cmd[], int delta);
int read_trace2(void** mem, int* ver, int size, long ts[], int len[], long lbn[], int cmd[]); 


vscsi_version_t test_vscsi_version(void *trace);

#endif
