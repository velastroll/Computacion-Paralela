/*
 * Simulacion simplificada de bombardeo de particulas de alta energia
 *
 * Computacion Paralela (Grado en Informatica)
 * 2017/2018
 *
 * (c) 2018 Arturo Gonzalez Escribano
 *
 * Modificada por el Grupo 04 
 */
 #include<stdio.h>
 #include<stdlib.h>
 #include<math.h>
 #include<cuda.h>
 #include<cputils.h>
 #include"kernel.cu"
 
 #define PI	3.14159f
 #define MAX_THREADS_PER_BLOCK 1024
 
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
 
	 /*---------------------------------------------------------------------*/
	 /* COMIENZO: No optimizar/paralelizar el main por encima de este punto */
 
 
 
	 /*Indicamos la GPU (DEVICE) que vamos a utilizar*/
	 float *d_layer;
	 float *d_layerCopy;
	 int *d_pos;
	 float *d_max;
 
	 /* 3. Reservar memoria para las capas e inicializar a cero */
	 float *layer = (float *)malloc( sizeof(float) * layer_size );
	 float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
	 if ( layer == NULL || layer_copy == NULL ) {
		 fprintf(stderr,"Error: Allocating the layer memory\n");
		 exit( EXIT_FAILURE );
	 }
 
	 /* Variable para controlar los errores */
	 cudaError_t error;
 
 
	 /* Reservamos la memoria de la GPU */
	 
	 // layer
	 error = cudaMalloc((void**) &d_layer, (float) layer_size*sizeof(float));
	 if (error != cudaSuccess) printf("Error CUDA: %s \n", cudaGetErrorString(error));
 
	 // layer_copy
	 error = cudaMalloc((void**) &d_layerCopy, (float) layer_size*sizeof(float));
	 if (error != cudaSuccess) printf("Error CUDA: %s \n", cudaGetErrorString(error));
 
	 // valor del maximo
	 error = cudaMalloc((void**) &d_max, (float) num_storms*sizeof(float));
	 if (error != cudaSuccess) printf("Error CUDA: %s \n", cudaGetErrorString(error));
 
	 // posiciones del maximo
	 error = cudaMalloc((void**) &d_pos, (int) num_storms*sizeof(int));
	 if (error != cudaSuccess) printf("Error CUDA: %s \n", cudaGetErrorString(error));
 
	 /* Inicialización de vectores */
	 for(k=0 ; k<layer_size ; k++) {
		 layer[k] = 0.0f;
		 layer_copy[k] = 0.0f;
	 }
 
	 /* 4. Fase de bombardeos */
	 for(i=0 ; i<num_storms ; i++) {
 
		 /* 4.1. Suma energia de impactos */
		 /* Para cada particula */
 
		 /* Copia de datos del HOST al DEVICE */
		 // copiamos layer a GPU
		 error = cudaMemcpy(d_layer, layer, layer_size * sizeof(float), cudaMemcpyHostToDevice);
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
		 // copiamos layer_copy en GPU
		 error = cudaMemcpy(d_layerCopy, layer_copy, layer_size * sizeof(float), cudaMemcpyHostToDevice);
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 // tamaños de los bloques y grid
		 dim3 nThreads(MAX_THREADS_PER_BLOCK);
		 dim3 nBlocks((layer_size+nThreads.x-1)/nThreads.x);
 
		 // funcion actualiza
		 for( j=0; j<storms[i].size; j++ ) {
			 float energia = (float)storms[i].posval[j*2+1] / 1000;
			 int posicion = storms[i].posval[j*2];
			 gpuActualiza<<<nThreads, nBlocks>>>(d_layer, posicion, energia);
		 }
 
		 /* 4.2. Relajacion entre tormentas de particulas */
		 /* 4.2.1. Copiar valores a capa auxiliar */
		 /* Función del kernel copiarLayer */
		 gpuCopiarLayer<<<nThreads, nBlocks>>>(d_layer, d_layerCopy);
 
		 /* 4.2.2. Actualizamos la capa sin contar los extremos */
		 // Atenuacion
		 gpuAtenuacion<<<nThreads, nBlocks>>>(d_layer, d_layerCopy, layer_size);
 
		 /* 4.3. Localizamos maximo */
 
		 /* enviamos los datos de maximos al gpu */
 
		 error = cudaMemcpy(d_max, maximos, num_storms*sizeof(float), cudaMemcpyHostToDevice);
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 error = cudaMemcpy(d_pos, posiciones, num_storms*sizeof(int), cudaMemcpyHostToDevice);
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 /* Calculamos los maximos */
 
		 gpuMaximos<<<nThreads, nBlocks>>>(d_layer, d_pos, d_max, layer_size, i);
 
		 /* traemos los datos al host de nuevo */
 
		 error = cudaMemcpy(layer_copy, d_layerCopy, layer_size*sizeof(float), cudaMemcpyDeviceToHost );
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 error = cudaMemcpy(layer, d_layer, layer_size*sizeof(float), cudaMemcpyDeviceToHost );
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 error = cudaMemcpy(maximos, d_max, num_storms*sizeof(float), cudaMemcpyDeviceToHost );
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
		 error = cudaMemcpy(posiciones, d_pos, num_storms*sizeof(int), cudaMemcpyDeviceToHost );
		 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
	 }
 
	 /* liberamos la memoria en el gpu */
 
	 error = cudaFree(d_layer);
	 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
	 error = cudaFree(d_layerCopy);
	 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
	 error = cudaFree(d_max);
	 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
	 error = cudaFree(d_pos);
	 if (error != cudaSuccess) printf("ErrCUDA: %s\n", cudaGetErrorString(error));
 
	 /*----------------------------------------------------------*/
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
 