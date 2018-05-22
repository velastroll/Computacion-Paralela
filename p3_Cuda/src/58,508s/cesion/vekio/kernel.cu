/* cada hilo copia su parte */
__global__ void gpuCopiarLayer(float *layer, float *layer_copy) {
	int idBloque = blockIdx.x + blockIdx.y*gridDim.x;
	int idGlobal = (idBloque*blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
	layer_copy[idGlobal]=layer[idGlobal];
}

/* se actualiza la capa en función de la energia, la posición y la capa anterior */
__global__ void gpuActualiza(float *layer, int posicion, float energy) {

	int idBloque = blockIdx.x + blockIdx.y * gridDim.x;
	int idGlobal = (idBloque*blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
	int distancia = posicion - idGlobal;

	if ( distancia < 0 ) distancia = - distancia;

	/* 2. El punto de impacto tiene distancia 1 */
	distancia = distancia + 1;

	/* 3. Raiz cuadrada de la distancia */
	float atenuacion = sqrtf( (float)distancia );

	/* 4. Calcular energia atenuada */
	float energia_k = energy / atenuacion;

	/* 5. No sumar si el valor absoluto es menor que umbral */
	if ( energia_k >= 0.001f || energia_k <= -0.001f )
		layer[idGlobal] = layer[idGlobal] + energia_k;
}

/* Actualizamos la capa sin contar los extremos */
__global__ void gpuAtenuacion(float *layer, float *layer_copy, int layer_size) {
	/*Formula para calcular la posicion*/
	int idBloque = blockIdx.x + blockIdx.y * gridDim.x;
	int idGlobal = (idBloque*blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
	if(idGlobal > 0 && idGlobal < layer_size){
		layer[idGlobal] = ( layer_copy[idGlobal-1] + layer_copy[idGlobal] + layer_copy[idGlobal+1] ) / 3;
	}
}


/* el valor de i es el de la reducion */
__global__ void gpuMaximos(float *layer, int *posiciones, float *maximos,  int layer_size, int i) {
	/*Formula para calcular la posicion*/
	int idBloque = blockIdx.x + blockIdx.y * gridDim.x;
	int idGlobal = (idBloque*blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;

	if(idGlobal > 0 && idGlobal < layer_size){
		if (layer[idGlobal] > layer[idGlobal-1] && layer[idGlobal] > layer[idGlobal+1]) {
			if (layer[idGlobal] > maximos[i] ) {
				maximos[i] = layer[idGlobal];
				posiciones[i] = idGlobal;
			}
		}
	}
}

