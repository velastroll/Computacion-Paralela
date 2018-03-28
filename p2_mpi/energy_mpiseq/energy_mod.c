/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 * Version: 2.0 (Atenuacion no lineal)
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cputils.h>
#include<mpi.h>

#define PI	3.14159f
#define UMBRAL	0.001f
#define ROOT_RANK 0

/* Estructura para almacenar los datos de una tormenta de particulas */
typedef struct {
	int size;
	int *posval;
} Storm;

/* FUNCIONES AUXILIARES: No se utilizan dentro de la medida de tiempo, dejar como estan */
/* Funcion de DEBUG: Imprimir el estado de la capa */
void debug_print(int layer_size, float *layer, int *posiciones, float *maximos, int num_storms ) {
	int i,k;
	if ( layer_size <= 35 ) {
		/* Recorrer capa */
		for( k=0; k<layer_size; k++ ) {
			/* Escribir valor del punto */
			printf("%10.4f |", layer[k] );

			/* Calcular el numero de caracteres normalizado con el maximo a 60 */
			int ticks = (int)( 60 * layer[k] / maximos[num_storms-1] );

			/* Escribir todos los caracteres menos el ultimo */
			for (i=0; i<ticks-1; i++ ) printf("o");

			/* Para maximos locales escribir ultimo caracter especial */
			if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
				printf("x");
			else
				printf("o");

			/* Si el punto es uno de los maximos especiales, annadir marca */
			for (i=0; i<num_storms; i++) 
				if ( posiciones[i] == k ) printf(" M%d", i );

			/* Fin de linea */
			printf("\n");
		}
	}
}

/*
 * Funcion: Lectura de fichero con datos de tormenta de particulas
 */
Storm read_storm_file( char *fname ) {
	FILE *fstorm = cp_abrir_fichero( fname );
	if ( fstorm == NULL ) {
		fprintf(stderr,"Error: Opening storm file %s\n", fname );
		exit( EXIT_FAILURE );
	}

	Storm storm;	
	int ok = fscanf(fstorm, "%d", &(storm.size) );
	if ( ok != 1 ) {
		fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
		exit( EXIT_FAILURE );
	}

	storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
	if ( storm.posval == NULL ) {
		fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
		exit( EXIT_FAILURE );
	}
	
	int elem;
	for ( elem=0; elem<storm.size; elem++ ) {
		ok = fscanf(fstorm, "%d %d\n", 
					&(storm.posval[elem*2]),
					&(storm.posval[elem*2+1]) );
		if ( ok != 2 ) {
			fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
			exit( EXIT_FAILURE );
		}
	}
	fclose( fstorm );

	return storm;
}

/*
 * PROGRAMA PRINCIPAL
 */
int main(int argc, char *argv[]) {
	int i,j,k;
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc<3) {
		if (rank == ROOT_RANK)
			fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
		exit( EXIT_FAILURE );
	}

	int layer_size = atoi( argv[1] );
	int num_storms = argc-2;
	Storm storms[ num_storms ];

	for( i=2; i<argc; i++ ) 
		storms[i-2] = read_storm_file( argv[i] );

	float maximos[ num_storms ];
	int posiciones[ num_storms ];
	for (i=0; i<num_storms; i++) {
		maximos[i] = 0.0f;
		posiciones[i] = 0;
	}

	/* 2. Inicia medida de tiempo */
	MPI_Barrier(MPI_COMM_WORLD);
	double ttotal = cp_Wtime();

	/* ------------------------------------------------------------------- */
	/* COMIENZO: No optimizar/paralelizar el main por encima de este punto */

	/* 3. Reservar memoria para las capas e inicializar a cero */
	float *layer = (float *)malloc( sizeof(float) * layer_size );
	float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
	float maximo_global;
	if ( layer == NULL || layer_copy == NULL ) {
		fprintf(stderr,"Error: Allocating the layer memory\n");
		exit( EXIT_FAILURE );
	}
	for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
	for( k=0; k<layer_size; k++ ) layer_copy[k] = 0.0f;
	
	/* 4. Fase de bombardeos */
	for( i=0; i<num_storms; i++) {
		int amplitud = storms[i].size / size;
		int particion = layer_size / size;
		int inicio = rank*amplitud;
		int fin = inicio + amplitud;

		MPI_Barrier(MPI_COMM_WORLD);

		for( j=0; j<storms[i].size; j++ ) {
			int posicion = storms[i].posval[j*2];
			int pos_max;
			float energia = (float)storms[i].posval[j*2+1] / 1000;

			for( k=0; k<layer_size; k++ ) {
				/* Actualizar posicion */
				int distancia = posicion - k;
				if ( distancia < 0 ) distancia = - distancia;
				distancia = distancia + 1;
				float atenuacion = sqrtf( (float)distancia );
				float energia_k = energia / atenuacion;
				if ( energia_k >= UMBRAL || energia_k <= -UMBRAL )
					layer[k] = layer[k] + energia_k;
			}
		}
		for( k=0; k<layer_size; k++ ){ 
			layer[k] = layer_copy[k];
		}
		for( k=1; k<layer_size-1; k++ )
			layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

		/* 4.3. Localizar maximo */

		/* Cada proceso calcula los maximos de su particion */
		for( k=1+particion*rank; k<(particion*(rank+1)-1); k++ ) {
			if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
				if ( layer[k] > maximos[i] ) {
					maximos[i] = layer[k];
					posiciones[i] = k;
				}
			}
		} //end for each particle in storm

		MPI_Allreduce( &maximos[i], &maximo_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		if (maximos[i] == maximo_global){
			/* Seleccionamos donde ha sido el máximo */
			MPI_Bcast( &posiciones[i], size, MPI_INT, rank, MPI_COMM_WORLD );
		}
		
	} //end foreach storm

	/* -------------------------------------------------------- */
	/* FINAL: No optimizar/paralelizar por debajo de este punto */
	MPI_Barrier(MPI_COMM_WORLD);
	ttotal = cp_Wtime() - ttotal;

	#ifdef DEBUG
	debug_print( layer_size, layer, posiciones, maximos, num_storms );
	#endif

	if (rank == ROOT_RANK)
	{
		printf("\n");
		printf("Time: %lf\n", ttotal );
		printf("Result:");
		for (i=0; i<num_storms; i++)
			printf(" %d %f", posiciones[i], maximos[i] );
		printf("\n");
	}
	for( i=0; i<argc-2; i++ )
		free( storms[i].posval );

	MPI_Finalize();
	return 0;
}
