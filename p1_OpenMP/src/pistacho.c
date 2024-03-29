/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 * Version: 2.0 (Atenuacion no lineal)
 * 
 * Estudiantes del grupo 4:
 * - Alejandro Sanz
 * - Alvaro Velasco
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cputils.h>
#include<omp.h>

#define PI	3.14159f
#define UMBRAL	0.001f
typedef struct {
	int size;
	int *posval;
} Storm;

void debug_print(int layer_size, float *layer, int *posiciones, float *maximos, int num_storms ) {
	int i,k;
	if ( layer_size <= 35 ) {
		for( k=0; k<layer_size; k++ ) {
			printf("%10.4f |", layer[k] );
			int ticks = (int)( 60 * layer[k] / maximos[num_storms-1] );
			for (i=0; i<ticks-1; i++ ) printf("o");
			if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
				printf("x");
			else
				printf("o");
			for (i=0; i<num_storms; i++) 
				if ( posiciones[i] == k ) printf(" M%d", i );
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
	double ttotal = cp_Wtime();

	/* --------------------------------------------------------------------- *
	 * COMIENZO: No optimizar/paralelizar el main por encima de este punto   *
	 *-----------------------------------------------------------------------*/

	/* 3. Reservar memoria para las capas e inicializar a cero */
	float *layer = (float *)malloc( sizeof(float) * layer_size );
	float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
	if ( layer == NULL || layer_copy == NULL ) {
		fprintf(stderr,"Error: Allocating the layer memory\n");
		exit( EXIT_FAILURE );
	}

	for( k=0; k<layer_size; k++ ){
		layer[k] = 0.0f;
		layer_copy[k] = 0.0f;
		}

	/* 4. Fase de bombardeos */
	
	/* Bucle que no conseguimos paralelizar debido al 4.3 */
	for( i=0; i<num_storms; i++) {

		/* 4.1. Suma energia de impactos */
		/* Para cada particula */		
		for( j=0; j<storms[i].size; j++ ) {
		
			float energia = (float)storms[i].posval[j*2+1] / 1000;
			int posicion = storms[i].posval[j*2];

			/* ------- PARALELIZAMOS LA FUNCION ACTUALIZA. -------- */
			#pragma omp parallel for firstprivate(energia, posicion)
			for( k=0; k<layer_size; k++ ) {
				int distancia = posicion - k;

				if ( distancia < 0 ) {
					distancia = - distancia;
				}

				/* 4.1.2. El punto de impacto tiene distancia 1 */
				distancia = distancia + 1;
				/* 4.1.3. Raiz cuadrada de la distancia */
				float atenuacion = sqrtf( (float)distancia );

				/* 4.1.4. Calcular energia atenuada */
				float energia_k = energia / atenuacion;

				/* 4.1.5. No sumar si el valor absoluto es menor que umbral */
				if ( energia_k >= UMBRAL || energia_k <= -UMBRAL )
					layer[k] = layer[k] + energia_k;
			} /* --------------------------------------------------- */
		}

		/* Paralelizado este bucle y el de abajo, el tiempo baja */
		#pragma omp parallel for shared(layer, layer_copy), firstprivate(layer_size)
		for( k=0; k<layer_size; k++ ) 
			layer_copy[k] = layer[k];

		#pragma omp parallel for shared(layer, layer_copy), firstprivate(layer_size)
		for( k=1; k<layer_size-1; k++ )
			layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;


		/*  4.3 : BUCLE QUE DA PROBELMAS PARALELIZAR */	
		#pragma omp parallel for			
		for( k=1; k<layer_size-1; k++ ) {
			/**
			 * Una reestructuacion de las comprobaciones:
			 * Primero que vea si es maximo, y luego que compruebe sus vecinos.
			 * Antes, estaba de forma: 	(B&C), A.
			 * Ahora esta de forma:		A, B, C.
			 **/
			#pragma omp critical
			{
				if ( layer[k] > maximos[i] ) {			// A			
					if ( layer[k] > layer[k-1] ) {		// B
						if ( layer[k] > layer[k+1] ) {  // C
							maximos[i] = layer[k];
							posiciones[i] = k;
						}
					}
				}
			}
		}
	}

	/*----- FINAL: No optimizar/paralelizar por debajo de este punto ----- */

	/* 5. Final de medida de tiempo */
	ttotal = cp_Wtime() - ttotal;

	/* 6. DEBUG: Dibujar resultado (Solo para capas con hasta 35 puntos) */
	#ifdef DEBUG
	debug_print( layer_size, layer, posiciones, maximos, num_storms );
	#endif

	/* 7. Salida de resultados para tablon */
	printf("\n");
	/* 7.1. Tiempo total de la computacion */
	printf("Time: %lf\n", ttotal );
	/* 7.2. Escribir los maximos */
	printf("Result:");
	for (i=0; i<num_storms; i++)
		printf(" %d %f", posiciones[i], maximos[i] );
	printf("\n");

	/* 8. Liberar recursos */	
	for( i=0; i<argc-2; i++ )
		free( storms[i].posval );

	/* 9. Final correcto */
	return 0;
}
