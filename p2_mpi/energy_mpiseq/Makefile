
CFLAGS=-O3 -I.
DEBUG_FLAGS=-g -DDEBUG
LIBS=-lm
CC=mpicc

energy: energy_modificado.c 
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
debug: energy_modificado.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $< -o $@ $(LIBS)

small:
	$ mpiexec -n 1 ./energy 20000 ./test/testSmall_20000

medium:
	$ mpiexec -n 1 ./energy 100000 ./test/testMedium_100000
	
clean:
	rm -f energy debug

