import numpy as np
import math
import sys
import time
from numba import cuda, jit, float64
import matplotlib.pyplot as plt

# ID del estudiante (fijo según tu ejemplo)
ID = 5

# Función a integrar (versión CPU)
def funcion_cpu(x):
    return x * x * x + math.sin(x) + math.cos(2 * x)

# Función a integrar (versión GPU)
@cuda.jit(device=True)
def funcion_gpu(x):
    return x * x * x + math.sin(x) + math.cos(2 * x)

# Método del trapecio secuencial en CPU
def trapecio_secuencial(a, b, n):
    h = (b - a) / n
    resultado = 0.5 * (funcion_cpu(a) + funcion_cpu(b))
    
    for i in range(1, n):
        x = a + i * h
        resultado += funcion_cpu(x)
    
    return h * resultado

# Kernel CUDA para el método del trapecio
@cuda.jit
def trapecio_kernel(a, h, n, resultado_parcial, resultado_final):
    """
    Kernel CUDA para calcular la integral por el método del trapecio.
    Cada hilo calcula un subconjunto de los puntos.
    """
    # Índice global del hilo
    idx = cuda.grid(1)
    
    # Número total de hilos
    total_threads = cuda.gridsize(1)
    
    # Calcular cuántos puntos calcula cada hilo
    puntos_por_hilo = n // total_threads
    resto = n % total_threads
    
    # Calcular inicio y fin para este hilo
    if idx < resto:
        inicio = idx * (puntos_por_hilo + 1)
        fin = inicio + (puntos_por_hilo + 1)
    else:
        inicio = idx * puntos_por_hilo + resto
        fin = inicio + puntos_por_hilo
    
    # Calcular la suma parcial para los puntos asignados
    suma_parcial = 0.0
    for i in range(inicio, fin):
        if i > 0 and i < n:  # Puntos interiores (no incluye extremos)
            x = a + i * h
            suma_parcial += funcion_gpu(x)
    
    # Guardar la suma parcial
    resultado_parcial[idx] = suma_parcial
    
    # Sincronizar todos los hilos
    cuda.syncthreads()
    
    # Reducción en el bloque (suma de todas las sumas parciales)
    if idx == 0:
        suma_total = 0.0
        for i in range(total_threads):
            suma_total += resultado_parcial[i]
        resultado_final[0] = suma_total

# Método del trapecio paralelo en GPU
def trapecio_gpu(a, b, n, threads_per_block=256, blocks_per_grid=64):
    """
    Calcula la integral usando GPU con CUDA.
    """
    h = (b - a) / n
    
    # Resultados parciales en GPU
    resultado_parcial_gpu = cuda.device_array(threads_per_block * blocks_per_grid, dtype=np.float64)
    resultado_final_gpu = cuda.device_array(1, dtype=np.float64)
    
    # Configurar grid y lanzar kernel
    trapecio_kernel[blocks_per_grid, threads_per_block](a, h, n, resultado_parcial_gpu, resultado_final_gpu)
    
    # Copiar resultado de GPU a CPU
    resultado_final_cpu = resultado_final_gpu.copy_to_host()
    
    # Calcular integral completa (incluyendo extremos)
    integral = h * (0.5 * (funcion_cpu(a) + funcion_cpu(b)) + resultado_final_cpu[0])
    
    return integral

# Kernel optimizado con memoria compartida
@cuda.jit
def trapecio_kernel_optimizado(a, h, n, resultado_final):
    """
    Kernel optimizado que usa memoria compartida para reducción.
    """
    # Memoria compartida para reducción
    shared = cuda.shared.array(256, dtype=float64)
    
    # Índices
    tid = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    total_threads = cuda.gridsize(1)
    
    # Calcular puntos por hilo
    puntos_por_hilo = n // total_threads
    resto = n % total_threads
    
    # Calcular inicio y fin
    if idx < resto:
        inicio = idx * (puntos_por_hilo + 1)
        fin = inicio + (puntos_por_hilo + 1)
    else:
        inicio = idx * puntos_por_hilo + resto
        fin = inicio + puntos_por_hilo
    
    # Calcular suma parcial
    suma_local = 0.0
    for i in range(inicio, fin):
        if i > 0 and i < n:  # Puntos interiores
            x = a + i * h
            suma_local += funcion_gpu(x)
    
    # Almacenar en memoria compartida
    shared[tid] = suma_local
    cuda.syncthreads()
    
    # Reducción en paralelo dentro del bloque
    i = cuda.blockDim.x // 2
    while i > 0:
        if tid < i:
            shared[tid] += shared[tid + i]
        cuda.syncthreads()
        i //= 2
    
    # El primer hilo del bloque escribe el resultado
    if tid == 0:
        resultado_final[cuda.blockIdx.x] = shared[0]

def trapecio_gpu_optimizado(a, b, n, threads_per_block=256, blocks_per_grid=64):
    """
    Versión optimizada con reducción en GPU.
    """
    h = (b - a) / n
    
    # Array para resultados de cada bloque
    resultados_bloque_gpu = cuda.device_array(blocks_per_grid, dtype=np.float64)
    
    # Lanzar kernel optimizado
    trapecio_kernel_optimizado[blocks_per_grid, threads_per_block](a, h, n, resultados_bloque_gpu)
    
    # Copiar resultados a CPU y sumar
    resultados_bloque_cpu = resultados_bloque_gpu.copy_to_host()
    suma_total = np.sum(resultados_bloque_cpu)
    
    # Calcular integral completa
    integral = h * (0.5 * (funcion_cpu(a) + funcion_cpu(b)) + suma_total)
    
    return integral

def analizar_speedup(a, b, n_max, output_file="speedup_resultados_cuda.txt"):
    """
    Analiza el speedup para diferentes números de intervalos.
    """
    print("=" * 50)
    print("ANÁLISIS DE SPEEDUP - INTEGRACIÓN CUDA")
    print("=" * 50)
    print(f"Límites: [{a:.2f}, {b:.2f}]")
    print(f"Número máximo de intervalos: {n_max}")
    print(f"ID del estudiante: {ID}")
    print("=" * 50)
    print()
    
    # Configuración GPU
    threads_per_block = 256
    blocks_per_grid = 64  # 256 * 64 = 16,384 hilos
    
    # Crear archivo de resultados
    with open(output_file, "w") as archivo:
        archivo.write("ANÁLISIS DE SPEEDUP - INTEGRACIÓN CUDA\n")
        archivo.write("=" * 50 + "\n")
        archivo.write(f"Límites: [{a:.2f}, {b:.2f}]\n")
        archivo.write(f"Threads por bloque: {threads_per_block}\n")
        archivo.write(f"Bloques por grid: {blocks_per_grid}\n")
        archivo.write(f"Total hilos: {threads_per_block * blocks_per_grid}\n")
        archivo.write(f"Función: f(x) = x³ + sin(x) + cos(2x)\n\n")
        archivo.write(f"{'Intervalos':<15} {'T.CPU(s)':<15} {'T.GPU(s)':<15} {'Speedup':<15} {'Eficiencia(%)':<15}\n")
        archivo.write(f"{'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}\n")
    
    # Probar diferentes números de intervalos (potencias de 10)
    for potencia in range(1, 11):
        n = int(10 ** potencia)
        
        if n > n_max:
            break
        
        print(f"Probando con n = {n} intervalos...")
        
        # Calcular tiempo CPU
        inicio_cpu = time.time()
        resultado_cpu = trapecio_secuencial(a, b, n)
        fin_cpu = time.time()
        tiempo_cpu = fin_cpu - inicio_cpu
        
        print(f"  Tiempo CPU: {tiempo_cpu:.6f} s")
        
        # Calcular tiempo GPU (incluye transferencia de datos)
        inicio_gpu = time.time()
        resultado_gpu = trapecio_gpu_optimizado(a, b, n, threads_per_block, blocks_per_grid)
        fin_gpu = time.time()
        tiempo_gpu = fin_gpu - inicio_gpu
        
        print(f"  Tiempo GPU: {tiempo_gpu:.6f} s")
        
        # Calcular speedup y eficiencia
        speedup = tiempo_cpu / tiempo_gpu if tiempo_gpu > 0 else 0
        # Eficiencia basada en número de núcleos CUDA disponibles
        # (estimación: T4 tiene 2560 núcleos CUDA)
        eficiencia = (speedup / 2560) * 100 if speedup > 0 else 0
        
        print(f"  Speedup: {speedup:.2f}")
        print(f"  Eficiencia: {eficiencia:.1f}%")
        print(f"  Resultado CPU: {resultado_cpu:.15f}")
        print(f"  Resultado GPU: {resultado_gpu:.15f}")
        print(f"  Diferencia: {abs(resultado_cpu - resultado_gpu):.15f}")
        print()
        
        # Escribir en archivo
        with open(output_file, "a") as archivo:
            archivo.write(f"{n:<15} {tiempo_cpu:<15.6f} {tiempo_gpu:<15.6f} {speedup:<15.2f} {eficiencia:<15.1f}\n")
    
    # Agregar información adicional al archivo
    with open(output_file, "a") as archivo:
        archivo.write("\n" + "=" * 50 + "\n")
        archivo.write("Leyenda:\n")
        archivo.write("- Speedup = Tiempo_CPU / Tiempo_GPU\n")
        archivo.write("- Eficiencia = (Speedup / 2560) * 100%\n")
        archivo.write("  (2560 es el número estimado de núcleos CUDA en Tesla T4)\n")
        archivo.write("- Tiempo GPU incluye transferencia de datos CPU↔GPU\n")
    
    print(f"Resultados guardados en '{output_file}'")
    print("=" * 50)
    print("ANÁLISIS COMPLETADO")
    print("=" * 50)

def generar_grafico_speedup(archivo_resultados="speedup_resultados_cuda.txt"):
    """
    Genera gráficos de speedup a partir del archivo de resultados.
    """
    try:
        # Leer datos del archivo
        intervalos = []
        tiempos_cpu = []
        tiempos_gpu = []
        speedups = []
        
        with open(archivo_resultados, "r") as f:
            lineas = f.readlines()
        
        # Buscar líneas con datos (omitir encabezados)
        for linea in lineas:
            if linea.strip() and not linea.startswith("=") and not linea.startswith("Leyenda") and not linea.startswith("ANÁLISIS") and not linea.startswith("Límites") and not linea.startswith("Threads") and not linea.startswith("Total") and not linea.startswith("Función") and not linea.startswith("Intervalos"):
                partes = linea.split()
                if len(partes) >= 5:
                    intervalos.append(int(partes[0]))
                    tiempos_cpu.append(float(partes[1]))
                    tiempos_gpu.append(float(partes[2]))
                    speedups.append(float(partes[3]))
        
        if not intervalos:
            print("No se encontraron datos para graficar")
            return
        
        # Crear figura con subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gráfico 1: Tiempos de ejecución
        axs[0, 0].plot(intervalos, tiempos_cpu, 'b-o', label='CPU', linewidth=2, markersize=8)
        axs[0, 0].plot(intervalos, tiempos_gpu, 'r-o', label='GPU', linewidth=2, markersize=8)
        axs[0, 0].set_xscale('log')
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_xlabel('Número de intervalos (log scale)')
        axs[0, 0].set_ylabel('Tiempo de ejecución (s, log scale)')
        axs[0, 0].set_title('Comparación de Tiempos de Ejecución')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Speedup
        axs[0, 1].plot(intervalos, speedups, 'g-o', linewidth=2, markersize=8)
        axs[0, 1].axhline(y=1, color='r', linestyle='--', label='Speedup = 1')
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_xlabel('Número de intervalos (log scale)')
        axs[0, 1].set_ylabel('Speedup')
        axs[0, 1].set_title('Speedup GPU vs CPU')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Aceleración relativa
        aceleracion = [t_cpu/t_gpu for t_cpu, t_gpu in zip(tiempos_cpu, tiempos_gpu)]
        axs[1, 0].plot(intervalos, aceleracion, 'm-o', linewidth=2, markersize=8)
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_xlabel('Número de intervalos (log scale)')
        axs[1, 0].set_ylabel('Aceleración')
        axs[1, 0].set_title('Aceleración Relativa')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Eficiencia
        eficiencias = [(s/2560)*100 for s in speedups]
        axs[1, 1].plot(intervalos, eficiencias, 'c-o', linewidth=2, markersize=8)
        axs[1, 1].axhline(y=100, color='r', linestyle='--', label='100% Eficiencia')
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_xlabel('Número de intervalos (log scale)')
        axs[1, 1].set_ylabel('Eficiencia (%)')
        axs[1, 1].set_title('Eficiencia del Paralelismo GPU')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('speedup_graficos_cuda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráficos generados y guardados como 'speedup_graficos_cuda.png'")
        
    except Exception as e:
        print(f"Error al generar gráficos: {e}")

def main():
    """
    Función principal del programa.
    """
    # Verificar argumentos de línea de comandos
    if len(sys.argv) != 4:
        print("Uso: python integral_cuda.py <lim_inf> <lim_sup> <intervalos_max>")
        print("<intervalos_max>: número máximo de intervalos (se probarán potencias de 10 hasta este valor)")
        print("\nEjemplo: python integral_cuda.py 0 1 1000000")
        sys.exit(1)
    
    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        n_max = int(sys.argv[3])
        
        # Verificar que b > a
        if b <= a:
            print("Error: El límite superior debe ser mayor que el límite inferior")
            sys.exit(1)
        
        # Verificar que n_max > 0
        if n_max <= 0:
            print("Error: El número máximo de intervalos debe ser positivo")
            sys.exit(1)
        
        # Ejecutar análisis de speedup
        analizar_speedup(a, b, n_max)
        
        # Generar gráficos
        generar_grafico_speedup()
        
    except ValueError as e:
        print(f"Error en los argumentos: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()