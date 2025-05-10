import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def configurar_peft(modelo, r=8, lora_alpha=32):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.
    
    Args:
        modelo: Modelo base
        r (int): Rango de adaptadores LoRA
        lora_alpha (int): Escala alpha para LoRA
    
    Returns:
        modelo: Modelo adaptado para fine-tuning
    """
    # Definir configuración LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Para modelos de lenguaje causal
        r=r,                          # Rango de los adaptadores
        lora_alpha=lora_alpha,        # Escala alpha
        lora_dropout=0.1,             # Dropout para regularización
        bias="none",                  # No adaptar los bias
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Capas a modificar (típicas para transformers)
    )
    
    # Aplicar la configuración PEFT al modelo
    modelo_peft = get_peft_model(modelo, lora_config)
    
    # Imprimir información de los parámetros entrenables
    modelo_peft.print_trainable_parameters()
    
    return modelo_peft

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.
    
    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    # Crear la carpeta si no existe
    import os
    os.makedirs(ruta, exist_ok=True)
    
    # Guardar solo los adaptadores para un modelo PEFT
    if hasattr(modelo, "save_pretrained"):
        modelo.save_pretrained(ruta)
        print(f"Modelo guardado en: {ruta}")
    
    # Guardar el tokenizador
    tokenizador.save_pretrained(ruta)
    print(f"Tokenizador guardado en: {ruta}")
    
    # Guardar información de configuración adicional si es necesario
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(f"{ruta}/config_info.txt", "w") as f:
        f.write(f"Modelo guardado el: {timestamp}\n")

def cargar_modelo_personalizado(ruta):
    """
    Carga un modelo personalizado desde una ruta específica.
    
    Args:
        ruta (str): Ruta del modelo
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    # Cargar el tokenizador
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    
    # Determinar si se está cargando un modelo PEFT o un modelo completo
    import os
    if os.path.exists(f"{ruta}/adapter_config.json"):
        # Es un modelo PEFT, primero cargar el modelo base
        # Nota: En producción, deberías especificar exactamente qué modelo base usar
        base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # O el modelo base que hayas usado
        modelo_base = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Cargar los adaptadores PEFT
        modelo = PeftModel.from_pretrained(modelo_base, ruta)
        print(f"Modelo PEFT cargado desde: {ruta}")
    else:
        # Es un modelo completo
        modelo = AutoModelForCausalLM.from_pretrained(
            ruta, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Modelo completo cargado desde: {ruta}")
    
    return modelo, tokenizador

class Chatbot:
    """
    Clase para manejar la lógica del chatbot.
    """
    def __init__(self, modelo, tokenizador):
        self.modelo = modelo
        self.tokenizador = tokenizador
        self.historico = []
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
    
    def generar_respuesta(self, entrada, historico=None):
        """
        Genera una respuesta del chatbot basada en la entrada y el histórico.
        
        Args:
            entrada (str): El mensaje del usuario
            historico (list, optional): Historial de conversación
            
        Returns:
            tuple: (respuesta, historico actualizado)
        """
        if historico is None:
            historico = self.historico
        
        # Preparar el historial de conversación en el formato adecuado
        # Este formato puede variar según el modelo que utilices
        prompt = self._formatear_entrada(entrada, historico)
        
        # Tokenizar el prompt
        inputs = self.tokenizador(prompt, return_tensors="pt")
        inputs = {k: v.to(self.modelo.device) for k, v in inputs.items()}
        
        # Generar la respuesta
        with torch.no_grad():
            outputs = self.modelo.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizador.eos_token_id
            )
        
        # Decodificar la respuesta
        respuesta = self.tokenizador.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Actualizar el histórico en formato compatible con Gradio (formato de mensajes)
        historico.append({"role": "user", "content": entrada})
        historico.append({"role": "assistant", "content": respuesta})
        
        return respuesta, historico
    
    def _formatear_entrada(self, entrada, historico):
        """
        Formatea la entrada y el histórico para el modelo.
        El formato específico dependerá del modelo que uses.
        """
        # Ejemplo para un modelo tipo Mistral Instruct
        formatted_prompt = "<s>"
        
        # Convertir de formato Gradio al formato de prompting si es necesario
        for mensaje in historico:
            if isinstance(mensaje, dict) and "role" in mensaje:
                # Formato nuevo de Gradio (mensajes)
                if mensaje["role"] == "user":
                    user_content = mensaje["content"]
                    # Si hay un mensaje siguiente del asistente, incluirlo
                    idx = historico.index(mensaje)
                    if idx + 1 < len(historico) and historico[idx + 1]["role"] == "assistant":
                        assistant_content = historico[idx + 1]["content"]
                        formatted_prompt += f"[INST] {user_content} [/INST] {assistant_content}"
            elif isinstance(mensaje, dict) and "user" in mensaje:
                # Formato antiguo (pre-deprecation)
                formatted_prompt += f"[INST] {mensaje['user']} [/INST] {mensaje['bot']}"
        
        formatted_prompt += f"[INST] {entrada} [/INST]"
        return formatted_prompt
    
    def limpiar_historico(self):
        """Limpia el histórico de conversación"""
        self.historico = []
        return []

# Interfaz web simple con Gradio
def crear_interfaz_web(chatbot):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.
    
    Args:
        chatbot: Instancia del chatbot
        
    Returns:
        gr.Interface: Interfaz de Gradio
    """
    # Definir la función de callback para procesar la entrada
    def responder(mensaje, history):
        # Agregamos un pequeño retraso para simular pensamiento
        import time
        time.sleep(0.5)
        
        # Convertir el histórico de Gradio a nuestro formato si es necesario
        converted_history = []
        if history:
            for exchange in history:
                if isinstance(exchange, list) and len(exchange) == 2:
                    # Formato antiguo de Gradio (tuplas)
                    user_msg, bot_msg = exchange
                    converted_history.append({"role": "user", "content": user_msg})
                    converted_history.append({"role": "assistant", "content": bot_msg})
                else:
                    # Ya está en el formato correcto
                    converted_history.append(exchange)
        
        # Generar respuesta
        respuesta_texto, _ = chatbot.generar_respuesta(mensaje, converted_history)
        
        # Devolver la respuesta para Gradio ChatInterface
        return respuesta_texto
    
    # Crear la interfaz de chat con el nuevo formato de mensajes
    interfaz = gr.ChatInterface(
        fn=responder,
        title="Chatbot Personalizado",
        description="Interactúa con este chatbot basado en un modelo de lenguaje personalizado.",
        examples=["Hola, ¿cómo estás?", 
                 "¿Puedes explicarme qué es el aprendizaje profundo?",
                 "Escribe un poema sobre la inteligencia artificial"],
        theme="soft",
        chatbot=gr.Chatbot(type='messages'),
        # Especificar type='messages' para el ChatInterface también
        type='messages'
    )
    
    return interfaz

# Función principal para el despliegue
def main_despliegue():
    # Ruta donde está guardado el modelo personalizado
    ruta_modelo = "mi_modelo_personalizado"
    
    try:
        # Verificar si existe la ruta del modelo personalizado
        import os
        if os.path.exists(ruta_modelo):
            print(f"Cargando modelo personalizado desde: {ruta_modelo}")
            modelo, tokenizador = cargar_modelo_personalizado(ruta_modelo)
        else:
            # Si no existe, cargar un modelo de respaldo directamente
            print(f"La ruta {ruta_modelo} no existe. Cargando modelo de respaldo...")
            # Pasar directamente al else-branch sin lanzar excepciones
            raise ValueError(f"No se encontró el modelo en {ruta_modelo}")
            
        # Crear instancia del chatbot
        mi_chatbot = Chatbot(modelo, tokenizador)
        
        # Crear la interfaz web
        interfaz = crear_interfaz_web(mi_chatbot)
        
        # Lanzar la interfaz web
        interfaz.launch(
            server_name="0.0.0.0",  # Accesible desde cualquier IP
            server_port=7860,       # Puerto predeterminado de Gradio
            share=True              # Crear un enlace público temporal (opcional)
        )
    except Exception as e:
        print(f"Error al desplegar el chatbot: {e}")
        print("Cargando modelo de respaldo desde Hugging Face...")
        
        # Cargar un modelo pequeño de respaldo desde Hugging Face
        try:
            # Usar un modelo pequeño como respaldo para demostración
            modelo_respaldo_id = "facebook/opt-125m"  # Un modelo muy pequeño (125M parámetros)
            
            tokenizador = AutoTokenizer.from_pretrained(modelo_respaldo_id)
            modelo = AutoModelForCausalLM.from_pretrained(
                modelo_respaldo_id,
                device_map="auto"
            )
            
            print(f"Modelo de respaldo cargado correctamente: {modelo_respaldo_id}")
            
            # Crear instancia del chatbot con el modelo de respaldo
            mi_chatbot = Chatbot(modelo, tokenizador)
            
            # Crear y lanzar la interfaz
            interfaz = crear_interfaz_web(mi_chatbot)
            
            # Silenciar advertencias innecesarias
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Lanzar la interfaz
            interfaz.launch(share=True)
            
        except Exception as e2:
            print(f"Error al cargar el modelo de respaldo: {e2}")
            print("No se pudo iniciar el chatbot. Verifica tu conexión o permisos.")
        # Aquí podrías implementar un plan B

if __name__ == "__main__":
    main_despliegue()