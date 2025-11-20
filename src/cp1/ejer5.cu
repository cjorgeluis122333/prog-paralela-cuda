#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "Error CUDA en %s (%d): %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define IDX2C(row, col, ld) ((row) * (ld) + (col))

__global__ void matMulNaive(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = A[IDX2C(row, k, K)];
            float b = B[IDX2C(k, col, N)];
            sum += a * b;
        }
        C[IDX2C(row, col, N)] = sum;
    }
}

void ejercicio5_matMul() {
    printf("\n===== EJERCICIO 5: Multiplicación de matrices =====\n");

    // ID fijo como se solicita
    int ID = 5;

    // Calcular dimensiones según las restricciones
    int M = 16 + (ID % 4);  // filas de A
    int K = 16 + (ID % 5);  // columnas de A / filas de B
    int N = 16 + (ID % 3);  // columnas de B

    printf("ID del estudiante: %d\n", ID);
    printf("Dimensiones: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error al reservar memoria en host.\n");
        exit(EXIT_FAILURE);
    }

    // Inicializar matrices con valores aleatorios que incluyan el ID
    srand(ID); // Semilla reproducible basada en el ID
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)(rand() % 100 + 1); // Valores entre 1-100
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)(rand() % 100 + 1);
    }

    // Insertar el ID en posiciones aleatorias
    int posA = rand() % (M * K);
    int posB = rand() % (K * N);
    h_A[posA] = (float)ID;
    h_B[posB] = (float)ID;

    printf("ID insertado en A[%d] y B[%d]\n", posA, posB);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Configuración del kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Ejecutando kernel con %dx%d bloques y %dx%d hilos por bloque\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Calcular función hash (suma total de elementos)
    double hash = 0.0;
    for (int i = 0; i < M * N; ++i) {
        hash += (double)h_C[i];
    }
    printf("Hash de la matriz resultante (suma total): %.2f\n", hash);

    // Liberar recursos
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Fin del ejercicio 5.\n\n");
}

// Implementaciones vacías para evitar errores de enlace
void ejercicio1_holaGPU() {
    printf("\n===== EJERCICIO 1: Hola Mundo CUDA =====\n");
    printf("Función no implementada en este ejemplo.\n\n");
}

void ejercicio2_vectorAdd() {
    printf("\n===== EJERCICIO 2: Suma de vectores =====\n");
    printf("Función no implementada en este ejemplo.\n\n");
}

void ejercicio3_scaleVector() {
    printf("\n===== EJERCICIO 3: Escalar vector =====\n");
    printf("Función no implementada en este ejemplo.\n\n");
}

void ejercicio4_maxReduction() {
    printf("\n===== EJERCICIO 4: Reducción máxima =====\n");
    printf("Función no implementada en este ejemplo.\n\n");
}

int main() {
  ejercicio5_matMul();

    return 0;
}
