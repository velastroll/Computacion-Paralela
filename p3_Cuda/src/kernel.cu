__global__ void gpu_Actualizar(float *layer, int posicion, float energia,int layer_size) {
	float umbral = 0.001;
	int gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	if(gid < layer_size){
		int distancia = posicion - gid;
		if ( distancia < 0 ) distancia = - distancia;
		distancia = distancia + 1;
		float atenuacion = sqrtf( (float)distancia );
		float energia_k = energia / atenuacion;
		if ( energia_k >= umbral || energia_k <= -umbral ) layer[gid] = layer[gid] + energia_k;
	}
}

__global__ void gpu_Copiar(float *layer, float *layer_copy,int layer_size) {	
	int gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	if(gid < layer_size) layer_copy[gid]=layer[gid];
}

__global__ void gpu_Relajacion(float *layer, float *layer_copy, int layer_size) {
	int gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	if(gid>0 && gid < layer_size-1) layer[gid] = ( layer_copy[gid-1] + layer_copy[gid] + layer_copy[gid+1] ) / 3;
}

__global__ void gpu_Maximo(float *maximo, int *posicion, float *layer, int layer_size){
	int k, gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	if (gid == 0){ 
		for( k=1; k<layer_size-1; k++ ) {
			if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
				if ( layer[k] > maximo[0] ) {
					maximo[0] = layer[k];
					posicion[0] = k;
				}
			}
		} 
	}
}
