class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """
    
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        """
        Inicializa el gestor de contexto.
        
        Args:
            longitud_maxima (int): Número máximo de tokens a mantener en el contexto
            formato_mensaje (callable): Función para formatear mensajes (por defecto, None)
        """
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado
        
    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
            
        Returns:
            str: Mensaje formateado
        """
        prefijos = {
            'sistema': '### Instrucciones del Sistema:\n',
            'usuario': '### Usuario:\n',
            'asistente': '### Asistente:\n'
        }
        
        prefijo = prefijos.get(rol.lower(), '### ')
        return f"{prefijo}{contenido}\n"
    
    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        mensaje = {
            'rol': rol.lower(),
            'contenido': contenido
        }
        self.historial.append(mensaje)
    
    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.
        
        Returns:
            str: Prompt completo para el modelo
        """
        prompt_completo = ""
        
        for mensaje in self.historial:
            prompt_completo += self.formato_mensaje(mensaje['rol'], mensaje['contenido'])
            
        return prompt_completo
    
    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.
        
        Args:
            tokenizador: Tokenizador del modelo
        """
        # Verificar la longitud actual del historial en tokens
        prompt_actual = self.construir_prompt_completo()
        tokens_actuales = len(tokenizador.encode(prompt_actual))
        
        # Si no excede la longitud máxima, no hacer nada
        if tokens_actuales <= self.longitud_maxima:
            return
        
        # Mantener el mensaje del sistema (si existe)
        mensajes_sistema = []
        otros_mensajes = []
        
        for mensaje in self.historial:
            if mensaje['rol'] == 'sistema':
                mensajes_sistema.append(mensaje)
            else:
                otros_mensajes.append(mensaje)
        
        # Eliminar mensajes más antiguos hasta que quepa en el límite
        while True:
            # Preservar al menos un intercambio (un mensaje de usuario y su respuesta)
            if len(otros_mensajes) <= 2:
                break
                
            # Eliminar el mensaje más antiguo (que no sea del sistema)
            otros_mensajes.pop(0)
            
            # Reconstruir el historial temporal y verificar
            historial_temporal = mensajes_sistema + otros_mensajes
            prompt_temporal = ""
            
            for mensaje in historial_temporal:
                prompt_temporal += self.formato_mensaje(mensaje['rol'], mensaje['contenido'])
                
            tokens_actuales = len(tokenizador.encode(prompt_temporal))
            
            # Si ahora cabe en el límite, actualizar el historial y salir
            if tokens_actuales <= self.longitud_maxima:
                self.historial = historial_temporal
                return
        
        # Si llegamos aquí y aún no cabe, truncar el contenido del mensaje más antiguo
        if len(otros_mensajes) > 1:
            mensaje_a_truncar = otros_mensajes[0]
            contenido_original = mensaje_a_truncar['contenido']
            
            # Truncar progresivamente hasta que quepa
            factor_truncado = 0.8  # Comienza eliminando el 20% del mensaje
            
            while factor_truncado > 0.1:  # No eliminar más del 90% del mensaje
                longitud_nueva = int(len(contenido_original) * factor_truncado)
                mensaje_a_truncar['contenido'] = contenido_original[:longitud_nueva] + "... [truncado]"
                
                # Reconstruir el historial y verificar
                historial_temporal = mensajes_sistema + otros_mensajes
                prompt_temporal = ""
                
                for mensaje in historial_temporal:
                    prompt_temporal += self.formato_mensaje(mensaje['rol'], mensaje['contenido'])
                    
                tokens_actuales = len(tokenizador.encode(prompt_temporal))
                
                # Si ahora cabe en el límite, actualizar el historial y salir
                if tokens_actuales <= self.longitud_maxima:
                    self.historial = historial_temporal
                    return
                    
                factor_truncado -= 0.1  # Reducir más si aún no cabe
        
        # Si todo lo anterior falla, mantener solo el mensaje del sistema y el último intercambio
        if len(otros_mensajes) >= 2:
            ultimos_dos = otros_mensajes[-2:]
            self.historial = mensajes_sistema + ultimos_dos


# Funciones auxiliares simuladas para completar el ejemplo
def cargar_modelo(modelo_id):
    """Simula la carga de un modelo y tokenizador."""
    # En una implementación real, usaría transformers
    class ModeloSimulado:
        def generate(self, *args, **kwargs):
            return ["Respuesta simulada"]
        
        def to(self, device):
            return self
    
    class TokenizadorSimulado:
        def encode(self, texto):
            # Simula tokenización por palabras
            return texto.split()
            
        def decode(self, tokens):
            if isinstance(tokens, list) and isinstance(tokens[0], str):
                return " ".join(tokens)
            return "Respuesta decodificada"
    
    return ModeloSimulado(), TokenizadorSimulado()

def verificar_dispositivo():
    """Simula la verificación del dispositivo."""
    return "cpu"  # En una implementación real: 'cuda' si hay GPU disponible


# Clase principal del chatbot
class Chatbot:
    """
    Implementación de chatbot con manejo de contexto.
    """
    
    def __init__(self, modelo_id, instrucciones_sistema=None):
        """
        Inicializa el chatbot.
        
        Args:
            modelo_id (str): Identificador del modelo en Hugging Face
            instrucciones_sistema (str): Instrucciones de comportamiento del sistema
        """
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        self.gestor_contexto = GestorContexto()
        
        # Inicializar el contexto con instrucciones del sistema
        if instrucciones_sistema:
            self.gestor_contexto.agregar_mensaje('sistema', instrucciones_sistema)
    
    def responder(self, mensaje_usuario, parametros_generacion=None):
        """
        Genera una respuesta al mensaje del usuario.
        
        Args:
            mensaje_usuario (str): Mensaje del usuario
            parametros_generacion (dict): Parámetros para la generación
            
        Returns:
            str: Respuesta del chatbot
        """
        # Parámetros de generación predeterminados
        if parametros_generacion is None:
            parametros_generacion = {
                'max_length': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            }
            
        # 1. Agregar mensaje del usuario al contexto
        self.gestor_contexto.agregar_mensaje('usuario', mensaje_usuario)
        
        # 2. Construir el prompt completo
        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        
        # 3. Verificar y truncar el contexto si es necesario
        self.gestor_contexto.truncar_historial(self.tokenizador)
        
        # 4. Reconstruir el prompt después de posible truncado
        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        
        # 5. Generar la respuesta (en una implementación real)
        # tokens_entrada = self.tokenizador.encode(prompt_completo, return_tensors="pt").to(self.dispositivo)
        # tokens_salida = self.modelo.generate(
        #     tokens_entrada,
        #     **parametros_generacion
        # )
        # respuesta = self.tokenizador.decode(tokens_salida[0], skip_special_tokens=True)
        
        # Simulación de respuesta para este ejemplo
        respuesta = f"Esta es una respuesta simulada al mensaje: '{mensaje_usuario}'"
        
        # 6. Agregar respuesta al contexto
        self.gestor_contexto.agregar_mensaje('asistente', respuesta)
        
        # 7. Devolver la respuesta
        return respuesta


# Prueba del sistema
def prueba_conversacion():
    # Crear una instancia del chatbot con instrucciones
    instrucciones = """
    Eres un asistente virtual amable y servicial. Responde de manera clara y concisa a las preguntas del usuario.
    Mantén un tono respetuoso y profesional en todo momento.
    """
    
    chatbot = Chatbot("gpt2", instrucciones_sistema=instrucciones)
    
    # Simular una conversación de varios turnos
    preguntas = [
        "Hola, ¿cómo estás?",
        "¿Puedes explicarme cómo funciona un gestor de contexto?",
        "¿Y cómo se implementa la truncación del historial?",
        "Gracias por la explicación. ¿Puedes resumir lo que hemos hablado?"
    ]
    
    print("=== Simulación de conversación ===")
    for pregunta in preguntas:
        print(f"\nUsuario: {pregunta}")
        respuesta = chatbot.responder(pregunta)
        print(f"Asistente: {respuesta}")
    
    print("\n=== Estado final del contexto ===")
    print(f"Número de mensajes en el historial: {len(chatbot.gestor_contexto.historial)}")
    print(f"Prompt completo:\n{chatbot.gestor_contexto.construir_prompt_completo()}")


# Ejecutar la prueba
if __name__ == "__main__":
    prueba_conversacion()