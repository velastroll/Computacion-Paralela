
__global__ void gpuLayer_copy(float *layer, float *layer_copy)
{

	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
	int gid = numbloqueengrid * hilosporbloque + numhiloenbloque;

	layer_copy[gid]=layer[gid];
}

__global__ void gpu_Actualiza(float *layer, int posicion, float energia,int layer_size)
{

	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
	int gid = numbloqueengrid * hilosporbloque + numhiloenbloque;

	/* Función actualiza */
	if(gid > 0 && gid < layer_size){
	int distancia = posicion - gid;
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

	if ( energia_k >= 0.001f || energia_k <= -0.001f )
		layer[gid] = layer[gid] + energia_k;
	}
}

__global__ void gpu_Extremos(float *layer, float *layer_copy, int layer_size)
{

	int hilosporbloque  = blockDim.x * blockDim.y;
	int numhiloenbloque = threadIdx.x + blockDim.x * threadIdx.y;
	int numbloqueengrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
	int gid = numbloqueengrid * hilosporbloque + numhiloenbloque;

	if(gid > 0 && gid < layer_size-1){
		layer[gid] = ( layer_copy[gid-1] + layer_copy[gid] + layer_copy[gid+1] ) / 3;
	}
}
