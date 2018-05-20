__global__ void gpuActualiza(float *layer, int posicion, float energia,int size) {
	float umbral = 0.001;
	int gid = ( blockIdx.x  + gridDim.x  * blockIdx.y ) * ( blockDim.x * blockDim.y ) + ( blockIdx.x  + gridDim.x  * blockIdx.y ) 

	if(gid < size) {
		int distancia = posicion - gid;
		if ( distancia < 0 ) distancia = - distancia;

		distancia = distancia + 1;
		float atenuacion = sqrtf( (float)distancia );
		float energia_k = energia / atenuacion;

		if ( energia_k >= umbral || energia_k <= -umbral )
			layer[gid] = layer[gid] + energia_k;
		
	}

	// nos aseguramos que se haya actualizado completamente la capa antes de calcular los máximos
	__syncthreads();
}

__global__ void gpuRelajacion(float *layer, float *layer_copy, int layer_size) {

	int gid = ( blockIdx.x  + gridDim.x  * blockIdx.y ) * ( blockDim.x * blockDim.y ) + ( blockIdx.x  + gridDim.x  * blockIdx.y ) 

	if(gid>0 && gid < layer_size-1){
		layer[gid] = ( layer_copy[gid-1] + layer_copy[gid] + layer_copy[gid+1] ) / 3;
	}
}

__global__ void gpuCopia(float *layer, float *layer_copy,int size) {

	int gid = ( blockIdx.x  + gridDim.x  * blockIdx.y ) * ( blockDim.x * blockDim.y ) + ( blockIdx.x  + gridDim.x  * blockIdx.y ) 

	if(gid < size){
		layer_copy[gid]=layer[gid];
	}
}
