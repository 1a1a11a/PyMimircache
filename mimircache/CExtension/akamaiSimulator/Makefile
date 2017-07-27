#define DEBUG_MAKEFILE 1

CC=gcc
CPP=g++


CFLAGS += -Wall -Winline 
#CFLAGS += -I/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m 
#macpro15 
#CFLAGS += -I/usr/local/Cellar/python3/3.6.0/Frameworks/Python.framework/Versions/3.6/include/python3.6m/

#CFLAGS += -I/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/numpy/core/include
CFLAGS += -I../headers/ -Iinclude/ -Iinclude/utils/ -Iinclude/libs -I../headers/cache -I/usr/lib/python3/dist-packages/numpy/core/include -I/usr/include/python3.4m/ -I/usr/lib/python3/dist-packages/numpy/core/include -I/usr/include/python3.5m/

# ifdef DEBUG_MAKEFILE 
CFLAGS+= -g -O0
#else
# CFLAGS+= -O2
#endif

CFLAGS += $(shell pkg-config --cflags glib-2.0)

CPPFLAGS += CFLAGS 
CPPFLAGS += -std=c++11  



LIBS    = $(shell pkg-config --libs glib-2.0 --libs gthread-2.0)
LIBS   += -lm 


CSOURCES = $(shell find . ! -name "*python*" ! -name "*deprecated*" -name "*.c") 
CPPSOURCES = $(shell find . ! -name "*python*" ! -name "*deprecated*" -name "*.cpp") 


COBJECTS := $(patsubst %.c, %.o, $(CSOURCES))
CPPOBJECTS := $(patsubst %.cpp, %.o, $(CPPSOURCES))


# HEADERS := $(wildcard include/*.h)
# HEADERS += $(wildcard include/*.hpp)
HEADERS := $(shell find . -name "*.h")
HEADERS += $(shell find . -name "*.hpp")

HEADERS += $(wildcard ../headers/*.h)
HEADERS += $(wildcard ../headers/cache/*.h)


$(info CSOURCES="$(CSOURCES)")
$(info CPPSOURCES="$(CPPSOURCES)")
# $(info HEADERS="$(HEADERS)")
$(info COBJECTS="$(COBJECTS)")
$(info CPPOBJECTS="$(CPPOBJECTS)")


# HEADERS = $(shell find . -name "*.h")


# OBJS+= main.o splay.o parda.o parda_print.o narray.o process_args.o seperate.o
# HEADERS= splay.h parda.h narray.h process_args.h seperate.h

# SOURCES=$(subst .o,.c, $(OBJS) )

# $(OBJS):$(HEADERS) makefile
.phony: all clean

all: akamaiSimulator 

akamaiSimulator: $(COBJECTS) $(CPPOBJECTS)
	$(CPP) $(CFLAGS) -o $@ $+ $(LIBS)

$(COBJECTS): %.o:%.c $(HEADERS) makefile
	$(CC) -c $(CFLAGS) $< -o $@ 

$(CPPOBJECTS): %.o:%.cpp $(HEADERS) makefile
	$(CPP) -c $(CFLAGS) $< -o $@ 

# %.d: %.c
# 	set -e; rm -f $@; \
# 	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
#         sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
#         rm -f $@.$$$$
# include $(sources:.c=.d

# $(info SOURCES="$(OBJECTS)")

clean:
	rm -f $(all) $(COBJECTS) $(CPPOBJECTS) akamaiSimulator

# 