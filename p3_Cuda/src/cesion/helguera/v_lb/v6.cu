/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"cputils.h"
#include"kernel.cu"

#define PI	3.14159f
#define numThreads 512

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

	/* 1.1. Leer argumentos */
	if (argc<3) {
		fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
		exit( EXIT_FAILURE );
	}

	int layer_size = atoi( argv[1] );
	int num_storms = argc-2;
	Storm storms[ num_storms ];

	/* 1.2. Leer datos de storms */
	for( i=2; i<argc; i++ )
		storms[i-2] = read_storm_file( argv[i] );

	/* 1.3. Inicializar maximos a cero */
	float maximos[ num_storms ];
	int posiciones[ num_storms ];
	for (i=0; i<num_storms; i++) {
		maximos[i] = 0.0f;
		posiciones[i] = 0;
	}

	/* 2. Inicia medida de tiempo */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

	 /* COMIENZO: No optimizar/paralelizar el main por encima de este punto */
	/////////////////////////////////////////////////////////////////////////


	/* 3. Reservar memoria para las capas e inicializar a cero */
	float *layer = (float *)malloc( sizeof(float) * layer_size );
	float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
	if ( layer == NULL || layer_copy == NULL ) {
		fprintf(stderr,"Error: Allocating the layer memory\n");
		exit( EXIT_FAILURE );
	}

	// Declaramos los futuros vectores de DEVICE y reservamos memoria
	float *d_layer;
	float *d_layer_copy;
	cudaMalloc((void**) &d_layer,(float) layer_size*sizeof(float));
	cudaMalloc((void**) &d_layer_copy,(float) layer_size*sizeof(float));

	/* Inicialización de vectores en HOST*/
	for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
	for( k=0; k<layer_size; k++ ) layer_copy[k] = 0.0f;

	// Inicializamos los de DEVICE copiando los de HOST
	cudaMemcpy(d_layer, layer, layer_size * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy(d_layer_copy, layer_copy, layer_size * sizeof(float), cudaMemcpyHostToDevice );

	for( i=0; i<num_storms; i++) {

		// Fase de golpeo de particulas
		for( j=0; j<storms[i].size; j++ ) {

			float energia = (float)storms[i].posval[j*2+1] / 1000;
			int posicion = storms[i].posval[j*2];

			gpuActualiza<<<MTHBLOQ, numThreads>>>(d_layer, posicion, energia,layer_size);
		}

		// paralelizamos el proceso de copia a layer_copy en DEVICE
		gpuCopia<<<MTHBLOQ, numThreads>>>(d_layer, d_layer_copy,layer_size);
		// Fase de relajación en DEVICE
		gpuRelajacion<<<MTHBLOQ, numThreads>>>(d_layer, d_layer_copy, layer_size);
		// TRraemos los datos de nuevo al HOST
		cudaMemcpy(layer_copy, d_layer_copy, layer_size*sizeof(float), cudaMemcpyDeviceToHost );
		cudaMemcpy(layer, d_layer, layer_size*sizeof(float), cudaMemcpyDeviceToHost );

		// Calculamos los maximos de manera secuencial en HOST
		for( k=1; k<layer_size-1; k++ ) {
			if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
				if ( layer[k] > maximos[i] ) {
					maximos[i] = layer[k];
					posiciones[i] = k;
				}
			}
		}
	}

	// Liberamos la memoria de DEVICE
	cudaFree(d_layer);
	cudaFree(d_layer_copy);


	 //////////////////////////////////////////////////////////////
	/* FINAL: No optimizar/paralelizar por debajo de este punto */

	/* 6. Final de medida de tiempo */
	cudaDeviceSynchronize();
	ttotal = cp_Wtime() - ttotal;

	/* 7. DEBUG: Dibujar resultado (Solo para capas con hasta 35 puntos) */
	#ifdef DEBUG
	debug_print( layer_size, layer, posiciones, maximos, num_storms );
	#endif

	/* 8. Salida de resultados para tablon */
	printf("\n");
	/* 8.1. Tiempo total de la computacion */
	printf("Time: %lf\n", ttotal );
	/* 8.2. Escribir los maximos */
	printf("Result:");
	for (i=0; i<num_storms; i++)
		printf(" %d %f", posiciones[i], maximos[i] );
	printf("\n");

	/* 9. Liberar recursos */
	for( i=0; i<argc-2; i++ )
		free( storms[i].posval );

	/* 10. Final correcto */
	return 0;
}
