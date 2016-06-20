DEBUG = 1

cc=gcc


CFLAGS += -Wall 
CFLAGS += -I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m 
CFLAGS += -I/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/numpy/core/include
CFLAGS += -Iheaders/ -Icache/ -Iheaders/cache
ifdef DEBUG
CFLAGS+= -g -O0
else
CFLAGS+= -O2
endif

CFLAGS += $(shell pkg-config --cflags glib-2.0)
LIBS    = $(shell pkg-config --libs glib-2.0 --libs gthread-2.0)
LIBS   += -lm

# OBJS = splay.o LRUAnalyzer.o 
# SOURCES := $(wildcard cache/*.c) 
# SOURCES += $(wildcard cacheReader/*.c) 
# SOURCES += $(wildcard profiler/generalProfiler/*.c) 
# SOURCES += $(wildcard cprofiler/*.c) 
# SOURCES += $(wildcard utils/*.c) 
# SOURCES += $(wildcard profiler/LRUProfiler/*.c) 

# SOURCES = $(filter-out *python*, $(SOURCES))

SOURCES = $(shell find . ! -name "*python*" -name "*.c")
# $(info SOURCES="$(SOURCES)")


OBJECTS := $(patsubst %.c, %.o, $(SOURCES))
HEADERS := $(wildcard headers/*.h)
HEADERS += $(wildcard headers/cache/*.h)
# HEADERS = $(shell find . -name "*.h")


# OBJS+= main.o splay.o parda.o parda_print.o narray.o process_args.o seperate.o
# HEADERS= splay.h parda.h narray.h process_args.h seperate.h

# SOURCES=$(subst .o,.c, $(OBJS) )

# $(OBJS):$(HEADERS) makefile
.phony: all clean

all: generalProfiler 

generalProfiler: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $+ $(LIBS)

$(info HEADERS="$(HEADERS)")

$(OBJECTS): %.o:%.c $(HEADERS) makefile
	$(CC) -c $(CFLAGS) $< -o $@ 
# %.d: %.c
# 	set -e; rm -f $@; \
# 	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
#         sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
#         rm -f $@.$$$$
# include $(sources:.c=.d

# $(info SOURCES="$(OBJECTS)")

clean:
	rm -f $(all) $(OBJECTS)
# $(shell find . -name "*.o" -exec rm {} \;)