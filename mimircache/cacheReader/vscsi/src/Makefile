CFLAGS=-m64 -Wall 
LFLAGS=
OBJS=trace_format.o trace_extract.o
MAIN=trace_extract

all: $(MAIN)

clean:
	rm $(MAIN) $(OBJS)

$(MAIN): $(OBJS)
	gcc $(CFLAGS) $(LFLAGS) -o $@ $(OBJS)

.c.o:
	gcc $(CFLAGS) -c -o $@ $<
	
