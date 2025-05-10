# Chatbot-con-Pytorch
# Respuestas Preguntas Teóricas:
1. Diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales

Los modelos de lenguaje se pueden clasificar en tres tipos principales según su arquitectura: encoder-only, decoder-only y encoder-decoder. Cada uno tiene características particulares que los hacen más adecuados para distintos casos de uso en chatbots.

Encoder-only: Estos modelos se centran exclusivamente en codificar la información de entrada para obtener una representación semántica rica. No generan texto, por lo que son útiles para tareas de comprensión del lenguaje como la clasificación de intenciones, la detección de entidades, análisis de sentimientos o búsqueda semántica. Un ejemplo de esta arquitectura es BERT.

Decoder-only: Utilizan solo un decodificador autoregresivo que genera texto una palabra a la vez, basándose en el contexto previo. Esta arquitectura es típica en modelos generativos como GPT. Son adecuados para tareas de generación de lenguaje natural fluida, como los chatbots conversacionales que deben responder de forma coherente y creativa.

Encoder-decoder: También conocidos como modelos seq2seq, utilizan un codificador para procesar la entrada y un decodificador que genera la salida basada en esa representación. Esta arquitectura es eficaz para tareas que requieren comprensión y generación de texto, como la traducción automática, el resumen de textos o chatbots orientados a tareas que requieren precisión y seguimiento de contexto. Ejemplos de este tipo de modelos incluyen T5 y BART.

Elección del modelo según el caso de uso:

Para generación de texto libre y natural, como en asistentes conversacionales abiertos, se recomienda un modelo decoder-only.

Para tareas que requieren tanto comprensión profunda como generación precisa (por ejemplo, traducción o respuestas basadas en contexto complejo), es más adecuado un modelo encoder-decoder.

Para tareas puramente analíticas o de clasificación, basta con un modelo encoder-only.

2. Concepto de "temperatura" en la generación de texto con LLMs

La "temperatura" es un parámetro que controla el nivel de aleatoriedad en la selección de palabras durante la generación de texto en modelos de lenguaje.

Una temperatura baja (por ejemplo, 0 o 0.2) hace que el modelo seleccione palabras con alta probabilidad, generando respuestas más deterministas, coherentes y conservadoras. Es útil para aplicaciones donde se requiere precisión y consistencia, como en chatbots médicos, jurídicos o técnicos.

Una temperatura alta (por ejemplo, 0.8 o 1.0) aumenta la diversidad de las respuestas al permitir que el modelo explore opciones menos probables. Esto puede generar respuestas más creativas, pero también más propensas a errores o incoherencias. Es adecuada para tareas como escritura creativa o generación de ideas.

Consideraciones al ajustar la temperatura:
La elección del valor de temperatura depende del objetivo del chatbot. Para tareas sensibles o que requieren factualidad, se recomienda una temperatura baja. Para aplicaciones que se beneficien de respuestas variadas o creativas, puede ser mejor usar una temperatura más alta.

3. Técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs

Las "alucinaciones" en modelos de lenguaje son respuestas que parecen coherentes pero contienen información falsa, imprecisa o inventada. Para mitigarlas, se pueden aplicar técnicas tanto en la etapa de inferencia como en la de diseño del prompt.

A. Técnicas a nivel de inferencia:

RAG (Retrieval-Augmented Generation): Se integran sistemas de recuperación de información para proporcionar al modelo acceso a datos reales (documentos, bases de conocimiento) que puede usar durante la generación.

Constrained decoding: Se limita el espacio de salida del modelo para evitar respuestas inadecuadas o inconsistentes.

Uso de temperatura baja y técnicas de muestreo como top-k o top-p: Esto reduce la aleatoriedad y mejora la coherencia.

Verificación cruzada: Se puede pedir al modelo que evalúe o revise sus respuestas antes de entregarlas.

B. Técnicas a nivel de prompt engineering:

Instrucciones explícitas: Formular indicaciones claras que obliguen al modelo a responder solo si tiene certeza o suficiente información.

Chain-of-Thought prompting: Incentivar al modelo a razonar paso a paso para llegar a respuestas más precisas.

Proporcionar contexto confiable: Incluir información verificable dentro del mismo prompt para que el modelo no dependa exclusivamente de su entrenamiento.
