import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Desactivar warning de symlinks en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Carpeta para cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "model_cache")

def cargar_modelo(nombre_modelo):
    tokenizador = AutoTokenizer.from_pretrained(
        nombre_modelo,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        padding_side="left",  # recomendado para decoder-only
    )
    # Si no hay pad_token, usar eos_token
    if tokenizador.pad_token_id is None:
        tokenizador.pad_token_id = tokenizador.eos_token_id

    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    modelo.eval()
    # Configurar pad_token_id globalmente
    modelo.generation_config.pad_token_id = tokenizador.pad_token_id
    if torch.cuda.is_available():
        modelo.half()
    return modelo, tokenizador

def verificar_dispositivo():
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device("cpu")
        print("GPU no disponible, usando CPU.")
    return dispositivo

def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")

    nombre_modelo = "PlanTL-GOB-ES/gpt2-base-bne"
    modelo, tokenizador = cargar_modelo(nombre_modelo)
    modelo.to(dispositivo)

    # 2. Tokenización con attention_mask
    prompt = "Hola, ¿cómo estás?"
    inputs = tokenizador(prompt, return_tensors="pt").to(dispositivo)

    # 3. Generación explícita con pad_token_id
    salida_ids = modelo.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizador.eos_token_id,
    )
    respuesta = tokenizador.decode(salida_ids[0], skip_special_tokens=True)
    print("Respuesta generada:", respuesta)

if __name__ == "__main__":
    main()
