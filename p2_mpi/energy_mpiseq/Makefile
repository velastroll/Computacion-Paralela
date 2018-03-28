
CFLAGS=-O3 -I.
DEBUG_FLAGS=-g -DDEBUG
LIBS=-lm
CC=mpicc

energy: energy_mod.c 
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

original: energy_original.c 
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

mediumoriginal:
	$ mpiexec -n 4 ./original 100000 ./test/testMedium_100000

debug: energy_mod.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $< -o $@ $(LIBS)

small:
	$ mpiexec -n 1 ./energy 20000 ./test/testSmall_20000

medium:
	$ mpiexec -n 5 ./energy 100000 ./test/testMedium_100000
	
clean:
	rm -f energy debug original
	clear

