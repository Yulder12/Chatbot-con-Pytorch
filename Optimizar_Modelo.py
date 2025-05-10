from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import time
import gc
import numpy as np
import os
import psutil
from typing import Dict, Tuple, Optional, List, Any

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.
    
    Args:
        bits (int): Bits para cuantización (4 u 8)
    
    Returns:
        BitsAndBytesConfig o None: Configuración de cuantización o None si no es posible
    """
    try:
        from transformers import BitsAndBytesConfig
        
        if bits not in [4, 8]:
            print(f"ADVERTENCIA: Valor de bits {bits} no soportado. Usando 8 bits.")
            bits = 8
        
        config_cuantizacion = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
        
        return config_cuantizacion
    
    except ImportError:
        print("ADVERTENCIA: bitsandbytes no está instalado. Instale con: pip install -U bitsandbytes")
        return None
    except Exception as e:
        print(f"Error al configurar cuantización: {e}")
        return None

def verificar_flash_attention():
    """
    Verifica si Flash Attention está disponible en el entorno.
    
    Returns:
        bool: True si está disponible, False en caso contrario
    """
    # Verificar si pytorch incluye flash attention
    try:
        if not torch.cuda.is_available():
            return False
            
        # Verificar versión de torch
        torch_version = torch.__version__
        if int(torch_version.split('.')[0]) < 2:
            return False
            
        # Intentar importar módulos relacionados
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            return True
            
        return False
    except:
        return False

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo con optimizaciones aplicadas.
    
    Args:
        nombre_modelo (str): Identificador del modelo
        optimizaciones (dict): Diccionario con flags para las optimizaciones
    
    Returns:
        tuple: (modelo, tokenizador)
    """
    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": False,  # Desactivado por defecto
            "bits": 8,              # Bits aumentados a 8
            "offload_cpu": False,
            "flash_attention": False,  # Desactivado por defecto
            "low_memory": True      # Nueva opción para entornos con poca memoria
        }
    
    # Limpiar caché para liberar memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Configurar el dispositivo
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilizando dispositivo: {dispositivo}")
    
    # Configurar opciones del modelo
    model_kwargs = {
        "low_cpu_mem_usage": optimizaciones.get("low_memory", True)
    }
    
    # Aplicar cuantización si está habilitada y disponible
    if optimizaciones.get("cuantizacion", False) and dispositivo == "cuda":
        quantization_config = configurar_cuantizacion(optimizaciones.get("bits", 8))
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            print(f"Aplicando cuantización de {optimizaciones.get('bits')} bits")
        else:
            print("No se pudo aplicar cuantización")
    elif optimizaciones.get("cuantizacion", False) and dispositivo == "cpu":
        print("Cuantización no disponible en CPU, se cargará el modelo completo")
    
    # Configurar offloading a CPU si está habilitado
    if optimizaciones.get("offload_cpu", False) and dispositivo == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = "offload_folder"
        print("Habilitando offloading a CPU")
    
    # Configurar atención eficiente (Flash Attention) si está habilitada
    if optimizaciones.get("flash_attention", False) and dispositivo == "cuda":
        if verificar_flash_attention():
            model_kwargs["use_flash_attention_2"] = True
            print("Habilitando Flash Attention 2")
        else:
            print("Flash Attention no disponible en este entorno")
    
    # Cargar modelo y tokenizador
    try:
        print(f"Cargando modelo {nombre_modelo}...")
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        
        # Asegurar que el tokenizador tenga un token de padding configurado
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token
        
        # Ajustar dtype según dispositivo
        dtype = torch.float16 if dispositivo == "cuda" else torch.float32
        
        # Cargar modelo con optimizaciones
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            torch_dtype=dtype,
            **model_kwargs
        )
        
        # Mover modelo al dispositivo si no se utiliza device_map="auto"
        if "device_map" not in model_kwargs:
            modelo = modelo.to(dispositivo)
        
        print(f"Modelo {nombre_modelo} cargado correctamente")
        
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise
    
    # Aplicar sliding window si está configurado
    if optimizaciones.get("sliding_window", False):
        window_size = optimizaciones.get("window_size", 1024)
        aplicar_sliding_window(modelo, window_size)
    
    # Aplicar optimizaciones específicas para CPU si estamos en ese entorno
    if dispositivo == "cpu" and optimizaciones.get("cpu_optimizations", True):
        aplicar_optimizaciones_cpu(modelo)
    
    return modelo, tokenizador

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.
    
    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    print(f"Configurando ventana deslizante con tamaño {window_size}")
    
    try:
        # Verificar si el modelo soporta configuración de ventana deslizante
        if hasattr(modelo.config, "sliding_window"):
            # Configurar directamente si el modelo lo soporta nativamente
            modelo.config.sliding_window = window_size
            print("Ventana deslizante configurada mediante config.sliding_window")
            return True
        elif hasattr(modelo.config, "window_size"):
            modelo.config.window_size = window_size
            print("Ventana deslizante configurada mediante config.window_size")
            return True
        elif hasattr(modelo.config, "attention_window"):
            # Para modelos tipo Longformer
            if isinstance(modelo.config.attention_window, int):
                modelo.config.attention_window = window_size
            else:
                modelo.config.attention_window = [window_size] * len(modelo.config.attention_window)
            print("Ventana deslizante configurada mediante config.attention_window")
            return True
        
        # Intentar configurar los módulos de atención individualmente
        configurado = False
        
        for nombre, modulo in modelo.named_modules():
            # Verificar si es un módulo de atención compatible
            if any(tipo in nombre.lower() for tipo in ["attention", "attn"]):
                if hasattr(modulo, "window_size"):
                    modulo.window_size = window_size
                    configurado = True
                elif hasattr(modulo, "max_positions"):
                    modulo.max_positions = window_size
                    configurado = True
                elif hasattr(modulo, "attention_window"):
                    modulo.attention_window = window_size
                    configurado = True
        
        if configurado:
            print("Ventana deslizante configurada a nivel de módulos de atención")
            return True
        else:
            print("ADVERTENCIA: No se pudo configurar la ventana deslizante, el modelo no parece soportarla")
            return False
    except Exception as e:
        print(f"Error al configurar ventana deslizante: {e}")
        return False

def aplicar_optimizaciones_cpu(modelo):
    """
    Aplica optimizaciones específicas para mejorar rendimiento en CPU.
    
    Args:
        modelo: Modelo a optimizar
    """
    try:
        # Optimizar operaciones de computación
        torch.set_num_threads(os.cpu_count())
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        # Intentar aplicar fusión de operaciones
        if hasattr(torch, 'jit'):
            # Intentar aplicar optimizaciones simples, pero manejar los casos en que el modelo no es compatible
            try:
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                # No convertimos todo el modelo a jit para evitar errores, solo habilitamos optimizaciones
            except:
                pass
        
        print("Optimizaciones CPU aplicadas")
        return True
    except Exception as e:
        print(f"Error al aplicar optimizaciones CPU: {e}")
        return False

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo, num_repeticiones=3):
    """
    Evalúa el rendimiento del modelo en términos de velocidad y memoria.
    
    Args:
        modelo: Modelo a evaluar
        tokenizador: Tokenizador del modelo
        texto_prueba (str): Texto para pruebas de rendimiento
        dispositivo: Dispositivo donde se ejecutará
        num_repeticiones (int): Número de repeticiones para promediar
    
    Returns:
        dict: Métricas de rendimiento
    """
    # Preparar entrada
    entradas = tokenizador(texto_prueba, return_tensors="pt", padding=True)
    
    # Mover entradas al dispositivo adecuado
    for key in entradas:
        entradas[key] = entradas[key].to(dispositivo)
    
    # Registrar uso de memoria antes de inferencia
    if dispositivo == "cuda":
        torch.cuda.synchronize()
        memoria_inicial = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    else:
        memoria_inicial = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    
    # Medir tiempo de inferencia
    tiempos = []
    
    # Calentar modelo (primera ejecución suele ser más lenta)
    with torch.no_grad():
        # Usar un número menor de tokens para el calentamiento para ahorrar tiempo
        _ = modelo.generate(**entradas, max_new_tokens=5)
    
    # Realizar mediciones reales
    for _ in range(num_repeticiones):
        # Limpiar cache entre repeticiones
        if dispositivo == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        inicio = time.time()
        with torch.no_grad():
            salida = modelo.generate(**entradas, max_new_tokens=20)
        
        if dispositivo == "cuda":
            torch.cuda.synchronize()
        
        fin = time.time()
        tiempos.append(fin - inicio)
    

    # Registrar uso de memoria después de inferencia
    if dispositivo == "cuda":
        torch.cuda.synchronize()
        memoria_final = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    else:
        memoria_final = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    
    # Calcular métricas
    tiempo_promedio = np.mean(tiempos)
    tiempo_std = np.std(tiempos)
    n_tokens_entrada = len(entradas["input_ids"][0])
    n_tokens_salida = len(salida[0]) - n_tokens_entrada
    tokens_por_segundo = n_tokens_salida / tiempo_promedio
    
    metricas = {
        "tiempo_inferencia_ms": tiempo_promedio * 1000,
        "desviacion_tiempo_ms": tiempo_std * 1000,
        "memoria_usada_mb": memoria_final - memoria_inicial,
        "tokens_por_segundo": tokens_por_segundo,
        "tokens_entrada": n_tokens_entrada,
        "tokens_generados": n_tokens_salida
    }
    
    return metricas

def imprimir_metricas(nombre_configuracion, metricas):
    """
    Imprime las métricas de rendimiento en un formato legible.
    
    Args:
        nombre_configuracion (str): Nombre de la configuración evaluada
        metricas (dict): Métricas de rendimiento
    """
    print(f"\n=== Métricas para {nombre_configuracion} ===")
    print(f"Tiempo de inferencia: {metricas['tiempo_inferencia_ms']:.2f} ms (±{metricas['desviacion_tiempo_ms']:.2f} ms)")
    print(f"Memoria usada: {metricas['memoria_usada_mb']:.2f} MB")
    print(f"Velocidad: {metricas['tokens_por_segundo']:.2f} tokens/segundo")
    print(f"Tokens de entrada: {metricas['tokens_entrada']}")
    print(f"Tokens generados: {metricas['tokens_generados']}")

# Función de demostración
def demo_optimizaciones():
    """
    Demuestra y compara diferentes configuraciones de optimización.
    """
    print("Iniciando demostración de optimizaciones...")
    
    # Modelo pequeño para pruebas (reemplazar con modelos más grandes en uso real)
    nombre_modelo = "distilgpt2"  # Un modelo pequeño para fines de demostración
    
    # Texto de prueba
    texto_prueba = """
    Este es un texto de ejemplo para evaluar el rendimiento del modelo en diferentes configuraciones.
    Las técnicas de optimización como la cuantización, el offloading y la atención de ventana deslizante
    pueden mejorar significativamente la eficiencia del modelo en dispositivos con recursos limitados.
    """
    
    # Determinar dispositivo disponible
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ejecutando pruebas en: {dispositivo}")
    
    # Lista para almacenar resultados
    resultados = []
    
    # Configuraciones adaptadas para CPU/GPU
    configuraciones = []
    
    # Configuraciones para CPU (más limitadas)
    if dispositivo == "cpu":
        configuraciones = [
            {
                "nombre": "Modelo base (sin optimizaciones)",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": False,
                    "offload_cpu": False,
                    "cpu_optimizations": False,
                    "low_memory": False
                }
            },
            {
                "nombre": "Modelo con ventana deslizante",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": True,
                    "window_size": 512,
                    "offload_cpu": False,
                    "cpu_optimizations": False,
                    "low_memory": False
                }
            },
            {
                "nombre": "Modelo con optimizaciones CPU",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": False,
                    "offload_cpu": False,
                    "cpu_optimizations": True,
                    "low_memory": False
                }
            },
            {
                "nombre": "Modelo con todas las optimizaciones para CPU",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": True,
                    "window_size": 512,
                    "offload_cpu": False,
                    "cpu_optimizations": True,
                    "low_memory": True
                }
            }
        ]
    else:
        # Configuraciones para GPU (todas las optimizaciones)
        configuraciones = [
            {
                "nombre": "Modelo base (sin optimizaciones)",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": False,
                    "offload_cpu": False
                }
            },
            {
                "nombre": "Modelo con cuantización de 8 bits",
                "optimizaciones": {
                    "cuantizacion": True,
                    "bits": 8,
                    "flash_attention": False,
                    "sliding_window": False,
                    "offload_cpu": False
                }
            },
            {
                "nombre": "Modelo con Flash Attention",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": True,
                    "sliding_window": False,
                    "offload_cpu": False
                }
            },
            {
                "nombre": "Modelo con ventana deslizante",
                "optimizaciones": {
                    "cuantizacion": False,
                    "flash_attention": False,
                    "sliding_window": True,
                    "window_size": 512,
                    "offload_cpu": False
                }
            },
            {
                "nombre": "Modelo con todas las optimizaciones",
                "optimizaciones": {
                    "cuantizacion": True,
                    "bits": 8,
                    "flash_attention": True,
                    "sliding_window": True,
                    "window_size": 512,
                    "offload_cpu": False
                }
            }
        ]
    
    # Evaluar cada configuración
    for config in configuraciones:
        try:
            print(f"\n\nEvaluando: {config['nombre']}")
            print("-" * 50)
            
            # Cargar modelo con la configuración específica
            modelo, tokenizador = cargar_modelo_optimizado(
                nombre_modelo, 
                optimizaciones=config["optimizaciones"]
            )
            
            # Número de repeticiones menor para CPU para evitar tiempos excesivos
            num_repeticiones = 2 if dispositivo == "cpu" else 3
            
            # Evaluar rendimiento
            metricas = evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo, num_repeticiones)
            
            # Guardar resultados
            resultados.append({
                "configuracion": config["nombre"],
                "metricas": metricas
            })
            
            # Mostrar métricas
            imprimir_metricas(config["nombre"], metricas)
            
            # Liberar memoria
            del modelo, tokenizador
            if dispositivo == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error al evaluar {config['nombre']}: {str(e)}")
    
    # Comparar resultados
    if resultados:
        print("\n\n=== Comparativa de Configuraciones ===")
        print("-" * 80)
        print(f"{'Configuración':<40} | {'Tiempo (ms)':<12} | {'Memoria (MB)':<12} | {'Tokens/s':<10}")
        print("-" * 80)
        
        for resultado in resultados:
            metricas = resultado["metricas"]
            print(f"{resultado['configuracion']:<40} | "
                f"{metricas['tiempo_inferencia_ms']:<12.2f} | "
                f"{metricas['memoria_usada_mb']:<12.2f} | "
                f"{metricas['tokens_por_segundo']:<10.2f}")
        
        # Identificar la mejor configuración (por tokens/segundo)
        mejor = max(resultados, key=lambda x: x["metricas"]["tokens_por_segundo"])
        print("\nMejor configuración por velocidad:")
        print(f"  {mejor['configuracion']} - {mejor['metricas']['tokens_por_segundo']:.2f} tokens/segundo")
        
        # Mejor por uso de memoria (menor es mejor)
        mejor_memoria = min(resultados, key=lambda x: x["metricas"]["memoria_usada_mb"])
        print("\nMejor configuración por uso de memoria:")
        print(f"  {mejor_memoria['configuracion']} - {mejor_memoria['metricas']['memoria_usada_mb']:.2f} MB")
    else:
        print("\nNo se pudieron completar pruebas suficientes para hacer una comparación")
    
    print("\nDemostración completada.")

# Si se ejecuta como script principal
if __name__ == "__main__":
    demo_optimizaciones()