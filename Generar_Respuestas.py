import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Desactivar warning de symlinks en Windows (opcional)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Carpeta para cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "model_cache")


def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador para generación causal.
    """
    tokenizador = AutoTokenizer.from_pretrained(
        nombre_modelo,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    if tokenizador.pad_token_id is None:
        tokenizador.pad_token_id = tokenizador.eos_token_id

    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    modelo.eval()
    # Configurar pad_token globalmente
    modelo.generation_config.pad_token_id = tokenizador.pad_token_id
    if torch.cuda.is_available():
        modelo.half()
    return modelo, tokenizador


def verificar_dispositivo():
    """
    Detecta si hay GPU o CPU y muestra información.
    """
    if torch.cuda.is_available():
        disp = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        disp = torch.device("cpu")
        print("GPU no disponible, usando CPU.")
    return disp


def preprocesar_entrada(texto, tokenizador, dispositivo, longitud_maxima=512):
    """
    Preprocesa el texto de entrada:
    - Tokeniza con padding y truncamiento
    - Añade attention_mask
    - Mueve tensores al dispositivo
    """
    entradas = tokenizador(
        texto,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=longitud_maxima,
    )
    entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
    return entradas


def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta dado el modelo y la entrada procesada.
    """
    # Parámetros por defecto
    if parametros_generacion is None:
        parametros_generacion = {
            "max_length": entrada_procesada["input_ids"].shape[1] + 50,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "num_return_sequences": 1,
            # pad_token_id ya está en generation_config
        }
    # Merge de parámetros
    gen_kwargs = {**parametros_generacion, **entrada_procesada}
    salida = modelo.generate(**gen_kwargs)
    respuesta = tokenizador.decode(salida[0], skip_special_tokens=True)
    return respuesta


def crear_prompt_sistema(instrucciones):
    """
    Construye un prompt de sistema para definir la personalidad o rol.
    """
    prompt = f"<sistema>\n{instrucciones}\n<usuario>:"
    return prompt


def interaccion_simple():
    """
    Ejemplo de interacción: define rol, procesa entrada, genera respuesta.
    """
    dispositivo = verificar_dispositivo()
    modelo, tokenizador = cargar_modelo("PlanTL-GOB-ES/gpt2-base-bne")
    modelo.to(dispositivo)

    # 1. Definir rol del chatbot
    instrucciones = (
        "Eres un asistente amable y profesional."
        " Responde en español de manera clara y concisa."
    )
    prompt_sistema = crear_prompt_sistema(instrucciones)

    # 2. Simular entrada del usuario
    texto_usuario = "Hola, ¿puedes explicarme qué es una red neuronal?"
    prompt = prompt_sistema + ' ' + texto_usuario

    # 3. Preprocesar
    entrada_proc = preprocesar_entrada(prompt, tokenizador, dispositivo)

    # 4. Generar respuesta
    respuesta = generar_respuesta(modelo, entrada_proc, tokenizador)
    print("Chatbot:", respuesta)


if __name__ == "__main__":
    interaccion_simple()