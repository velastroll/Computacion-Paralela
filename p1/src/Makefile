
CFLAGS=-fopenmp -O3 -I.
DEBUG_FLAGS=-g -DDEBUG
LIBS=-lm

intento: intento.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

best: g04.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

debug-best: g04.c
	gcc $(CFLAGS) $(DEBUG_FLAGS) $< -o $@ $(LIBS)

clean:
	rm -f energy debug intento best

