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


__global__ void gpu_reduceMaximo(float* g_candidatos, float* positions, int size){

	int gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	int s = size/2;
	if ( gid >= size/2) return;

	if(g_candidatos[ gid ] < g_candidatos[ gid + s]) {
		g_candidatos[ gid ] = g_candidatos[ s + gid  ];
		positions[gid] = positions[gid+s];
	}
	// Extra element
	if ( size%2 != 0 && gid == 0 ){
		if(g_candidatos[ 0 ] < g_candidatos[ size-1 ]) {
			g_candidatos[ 0 ] = g_candidatos[ size-1 ];
			positions[0] = size-1;
		}
	}

}


__global__ void gpu_obtenCandidatos (float *layer, float *candidatos, int layer_size ){

	int gid = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
	if (gid > layer_size) return;
	candidatos[gid] = 0;
	if (gid == 0 || gid == layer_size-1) return;
	if (layer[gid]>layer[gid-1] && layer[gid] > layer[gid+1]) candidatos[gid] = layer[gid];

}
