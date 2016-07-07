#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "vscsi_trace_format.h"
#include "reader.h"


int vscsi_setup(char *filename, READER* reader)
{
	struct stat st;	
	int f = -1;
	void *memT = NULL;
	
	if ( (f = open (filename, O_RDONLY)) < 0)
	{
		fprintf (stderr, "Unable to open '%s'\n", filename);
		return -1;
	}

	if ( (fstat (f, &st)) < 0)	
	{
		close (f);
		fprintf (stderr, "Unable to fstat '%s'\n", filename);
		return -1;
	}

	/* Version 2 records are the bigger of the two */
	if ((unsigned long long)st.st_size < (unsigned long long) sizeof(trace_v2_record_t) * MAX_TEST)
	{
		close (f);
		fprintf (stderr, "File too small, unable to read header.\n");
		return -1;
	}

	if ( (memT = (mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE, f, 0))) == MAP_FAILED)
	{
		close (f);
		fprintf (stderr, "Unable to allocate %llu bytes of memory\n", (unsigned long long) st.st_size);
		return -1;
	}

	reader->p = memT;
	reader->offset = 0;
	reader->ts = 0;
	reader->type = 'v';
    reader->data_type = 'l';


	vscsi_version_t ver = test_vscsi_version (memT);
	switch (ver)
	{
		case VSCSI1:
		case VSCSI2:
			reader->record_size = record_size(ver);
			reader->ver = ver;
			reader->total_num = st.st_size/(reader->record_size);
			break;
		
		case UNKNOWN:
		default:
			fprintf (stderr, "Trace format unrecognized.\n");
			return -1;
	}
	close(f);
	return 0;
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





// int main(int argc, char** argv){
// 	void ** mem = NULL; 
// 	int ver; 
// 	int delta; 
// 	int num_of_rec; 
// 	setup(argv[1], mem, &ver, &delta, &num_of_rec);
// 	int i;
// 	void* mem_original = *mem;
// 	for (i=0;i<10;i++)
// 		printf("%ld\n", read_trace(mem, &ver, &delta));
// 	finish(mem_original, (delta)*(num_of_rec));
// 	return 0;
// }
