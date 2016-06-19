DEBUG = 1

cc=gcc


CFLAGS += -Wall 
ifdef DEBUG
CFLAGS+= -g -O0
else
CFLAGS+= -O2
endif

CFLAGS += $(shell pkg-config --cflags glib-2.0)
LIBS    = $(shell pkg-config --libs glib-2.0 --libs gthread-2.0)

# OBJS = splay.o LRUAnalyzer.o 
SOURCES := $(wildcard *.c) 
OBJECTS := $(patsubst %.c, %.o, $(SOURCES))
HEADERS := $(wildcard *.h)

# OBJS+= main.o splay.o parda.o parda_print.o narray.o process_args.o seperate.o
# HEADERS= splay.h parda.h narray.h process_args.h seperate.h

# SOURCES=$(subst .o,.c, $(OBJS) )

# $(OBJS):$(HEADERS) makefile
$(OBJECTS): %.o:%.c $(HEADERS) makefile
	$(CC) -c $(CFLAGS) $< -o $@ 
%.d: %.c
	set -e; rm -f $@; \
	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
        sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
        rm -f $@.$$$$
include $(sources:.c=.d)


