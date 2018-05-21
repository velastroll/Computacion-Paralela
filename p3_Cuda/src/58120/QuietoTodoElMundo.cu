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
 #define HILOS 128
 
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
 
	  /////////////////////////////////////////////////////////////////////////
	 /* COMIENZO: No optimizar/paralelizar el main por encima de este punto */

	 /* 3. Reservar memoria para las capas e inicializar a cero */
	 float *layer;
	 *layer = cudaMallocHost(&layer,sizeof(float)*layer_size);
	 float *layer_copy = (float *)malloc( sizeof(float) * layer_size );
	 if ( layer == NULL || layer_copy == NULL ) {
		 fprintf(stderr,"Error: Allocating the layer memory\n");
		 exit( EXIT_FAILURE );
	 }
 
	// Iniciamos los vectores de HOST
	/*
	for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
	for( k=0; k<layer_size; k++ ) layer_copy[k] = 0.0f;
	 */
	 
	// Creamos la variable que comprobara los errores de CUDA
	cudaError_t err;

	// Reservamos e iniciamos la memoria necesaria en DEVICE
	float *d_layer;
	float *d_layerCopy;
	err = cudaMalloc((void**) &d_layer,(float) layer_size*sizeof(float));
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
	err = cudaMalloc((void**) &d_layerCopy,(float) layer_size*sizeof(float));
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
	err = cudaMemcpy(d_layer, layer, layer_size * sizeof(float), cudaMemcpyHostToDevice );
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
	err = cudaMemcpy(d_layerCopy, layer_copy, layer_size * sizeof(float), cudaMemcpyHostToDevice );
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
 
	int BLOQUES = (1+(layer_size-1)/HILOS);
	if (BLOQUES<1) BLOQUES=1;

	for( i=0; i<num_storms; i++) {

		// Actualizamos los golpes en la capa de DEVICE
		for( j=0; j<storms[i].size; j++ ) {
			float energia = (float)storms[i].posval[j*2+1] / 1000;
			int posicion = storms[i].posval[j*2];
			gpu_Actualizar<<<BLOQUES, HILOS>>>(d_layer, posicion, energia,layer_size);
		}
		
		// Hacemos una copia de la capa en DEVICE
		gpu_Copiar<<<BLOQUES, HILOS>>>(d_layer, d_layerCopy,layer_size);
		
		// Proceso de relajacion paralelizado en HOST
		gpu_Relajacion<<<BLOQUES, HILOS>>>(d_layer, d_layerCopy, layer_size);
 
		// Tramos el valor de la capa para calcular máximos
		err = cudaMemcpy(layer, d_layer, layer_size*sizeof(float), cudaMemcpyDeviceToHost );
		if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
 
		// Calculamos máximos en HOST => Hay que paralelizarlo
		for( k=1; k<layer_size-1; k++ ) {
			if ( layer[k] > maximos[i] ) {
				if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
					maximos[i] = layer[k];
					posiciones[i] = k;
				 }
			 }
		 }
	 }
 
	 //Liberamos memoria
	err = cudaFree(d_layer);
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);
	err = cudaFree(d_layerCopy);
	if (err != cudaSuccess) printf("CUDA-ERROR: %s\n", err);

	  //////////////////////////////////////////////////////////////
	 /* FINAL: No optimizar/paralelizar por debajo de este punto */
 
	 /* 6. Final de medida de tiempo */
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
 