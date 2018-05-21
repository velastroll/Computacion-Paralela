
__global__ void gpuFunc_copiarLayer(float *layer, float *layer_copy)
{
	/*Formula para calcular la posicion*/
	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalId = numbloqueengrid * hilosporbloque + numhiloenbloque;
	/*int bloqueId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalId = bloqueId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;*/
	layer_copy[globalId]=layer[globalId];
}

__global__ void gpuFunc_actualiza(float *layer, int posicion, float energia,int layer_size)
{
	/*Formula para calcular la posicion*/
	/*int bloqueId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalId = bloqueId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;*/
	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalId = numbloqueengrid * hilosporbloque + numhiloenbloque;
	/* Función actualiza */
	int distancia = posicion - globalId;
	if ( distancia < 0 ) distancia = - distancia;

	/* 2. El punto de impacto tiene distancia 1 */
	distancia = distancia + 1;

	/* 3. Raiz cuadrada de la distancia */
	//float atenuacion = (float)distancia*distancia;
	//float atenuacion = (float)distancia / PI;
	float atenuacion = sqrtf( (float)distancia );

	/* 4. Calcular energia atenuada */
	float energia_k = energia / atenuacion;

	/* 5. No sumar si el valor absoluto es menor que umbral */
	if(globalId > 0 && globalId < layer_size){
	if ( energia_k >= 0.001f || energia_k <= -0.001f )
		layer[globalId] = layer[globalId] + energia_k;
	}
}

__global__ void gpuFunc_extremos(float *layer, float *layer_copy, int layer_size)
{
	/*Formula para calcular la posicion*/
	/*int bloqueId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalId = bloqueId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;*/
	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalId = numbloqueengrid * hilosporbloque + numhiloenbloque;
	if(globalId > 0 && globalId < layer_size-1){
		layer[globalId] = ( layer_copy[globalId-1] + layer_copy[globalId] + layer_copy[globalId+1] ) / 3;
	}
}

__global__ void gpuFunc_maximos(float *layer, int *posiciones, float *maximos,  int layer_size, int i)
{
	/*Formula para calcular la posicion*/
	int bloqueId = blockIdx.x + blockIdx.y * gridDim.x;
	int globalId = bloqueId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if(globalId > 0 && globalId < layer_size){
		if ( layer[globalId] > layer[globalId-1] && layer[globalId] > layer[globalId+1] ) {
			if ( layer[globalId] > maximos[i] ) {
				maximos[i] = layer[globalId];
				posiciones[i] = globalId;
			}
		}
	}
}
