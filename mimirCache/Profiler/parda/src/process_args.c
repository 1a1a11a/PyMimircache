#include "h/process_args.h"

int is_omp = 0;
int is_mpi = 0;
int is_seperate = 0;
int is_binary = 0;
int threads = 1;
char inputFileName[200] = "d.in";
long lines = -1;
int buffersize = 10000;

int process_args(int argc, char **argv) {
  int c;
  is_omp = is_mpi = is_binary = 0;
  while (1) {
    static struct option long_options[] = {
      /* These options set a flag. */
      {"enable-omp", no_argument, 0, 'o'},
      {"enable-mpi",   no_argument, 0, 'm'},
      /* These options don't set a flag.
         We distinguish them by their indices. */
      {"fileformat",  required_argument, 0, 'f'},
      {"input",  required_argument, 0, 'i'},
      {"lines",  required_argument, 0, 'l'},
      {"threads",  required_argument, 0, 't'},
      {"enable-seperate",   no_argument, 0, 's'},
      {"help",   no_argument, 0, 'h'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "omb:",
        long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 's':
        is_seperate = 1;
        break;
      case 'o':
        is_omp = 1;
        break;

      case 'm':
        is_mpi = 1;
        break;

      case 'f':
        if(!strcmp(optarg, "binary")) is_binary = 1;
        else if(!strcmp(optarg,"text")) is_binary = 0;
        else printf("wrong value for fileformat. Try help\n"), abort();
        break;

      case 'i':
        strcpy(inputFileName, optarg);
        //if(stat ( inputFileName, &buf )!=0) printf("%s does not exist\n",inputFileName),abort();
        break;
      case 'l':
        lines = atol(optarg);
        break;
      case 't':
        threads = atol(optarg);
        break;
      case 'h':
        printf("case 1: seperate file\n");
        printf("./parda.x --enable-seperate --input=normal_137979.trace --lines=137979 --threads=4\n");
        printf("case 2: run with sequential algorithm\n");
        printf("./parda.x --input=normal_343684.trace --lines=343684\n");
        printf("case 3: run with OpenMp flag\n");
        printf("./parda.x --input=normal_343684.trace --lines=343684 --enable-omp --threads=4\n");
        printf("case 4: run with binary file input\n");
        printf("./parda.x --fileformat=binary --input=binary_167024.trace --lines=167024 > binary.re\n");
        exit(0);
        break;
      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
  {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    putchar ('\n');
  }
  if(lines == -1) printf("total lines number must be provided\n"),abort();
  return 0;
}
