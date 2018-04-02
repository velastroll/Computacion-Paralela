
CFLAGS=-O3 -I.
DEBUG_FLAGS=-g -DDEBUG
LIBS=-lm
CC=mpicc

energy: energy_mod.c 
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

original: energy_original.c 
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

debug: energy_mod.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $< -o $@ $(LIBS)

little1:
	$ mpiexec -n 4 ./energy 20 ./test/testMPI_20_4p_1.txt

little2:
	$ mpiexec -n 5 ./energy 20 ./test/testMPI_20_4p_2.txt

little3:
	$ mpiexec -n 3 ./energy 20 ./test/testMPI_20_4p_3.txt

small:
	$ mpiexec -n 1 ./energy 20000 ./test/testSmall_20000

medium:
	$ mpiexec -n 20 ./energy 100000 ./test/testMedium_100000

mix: 
	$ mpiexec -n 3 ./energy 100000 ./test/testMedium_100000 ./test/testSmall_20000 ./test/testMPI_20_4p_3.txt ./test/testMPI_20_4p_2.txt ./test/testMPI_20_4p_1.txt 

mix2: 
	$ mpiexec -n 3 ./energy 100000 ./test/testMedium_100000 ./test/testSmall_20000 ./test/testMedium_100000 ./test/testMPI_20_4p_3.txt ./test/testMPI_20_4p_2.txt ./test/testMedium_100000 ./test/testMPI_20_4p_1.txt 

mix3:
	$ mpiexec -n 20 ./original 20 ./test/testMPI_20_4p_1.txt  ./test/testMPI_20_4p_2.txt  ./test/testMPI_20_4p_3.txt

mix3mod:
	$ mpiexec -n 3 ./energy 20 ./test/testMPI_20_4p_1.txt  ./test/testMPI_20_4p_2.txt  ./test/testMPI_20_4p_3.txt


clean:
	rm -f energy debug original
	clear

