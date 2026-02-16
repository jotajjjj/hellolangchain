import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- HERRAMIENTAS ---

@tool
def leer_y_actualizar_archivo() -> str:
    """
    Lee el archivo base_datos.txt, identifica líneas sin comentario 
    y devuelve el contenido para que el agente procese las faltantes.
    """
    if not os.path.exists("base_datos.txt"):
        return "Error: El archivo base_datos.txt no existe."
    
    with open("base_datos.txt", "r", encoding="utf-8") as f:
        lineas = f.readlines()
    
    return "".join(lineas)

@tool
def guardar_archivo_final(contenido_completo: str) -> str:
    """Guarda el contenido ya procesado con los comentarios en el archivo."""
    with open("base_datos.txt", "w", encoding="utf-8") as f:
        f.write(contenido_completo)
    return "Archivo actualizado con éxito."

tools = [leer_y_actualizar_archivo, guardar_archivo_final]

# --- CONFIGURACIÓN MODELO 2.5 ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
    max_retries=3 # Ayuda con los errores de cuota temporales
).bind_tools(tools)

# --- LÓGICA DEL AGENTE ---

def assistant_node(state: MessagesState):
    instruccion = (
        "Eres un editor de bases de datos. Tu flujo de trabajo es:\n"
        "1. Usa 'leer_y_actualizar_archivo' para ver los datos.\n"
        "2. Identifica líneas con formato 'Nombre, Edad, Profesión' que NO tengan comentario.\n"
        "3. Para cada una, añade un comentario de máx 10 palabras sobre la profesión.\n"
        "4. Devuelve el archivo completo manteniendo el formato 'Nombre, Edad, Profesión, Comentario'.\n"
        "5. Usa 'guardar_archivo_final' para salvar los cambios."
    )
    # Concatenamos la instrucción de sistema con los mensajes
    messages = [("system", instruccion)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}

# --- CONSTRUCCIÓN DEL GRAFO ---
def router(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("assistant", assistant_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", router)
workflow.add_edge("tools", "assistant")

agent = workflow.compile()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("🚀 Iniciando procesamiento de base de datos...")
    
    # Solo una entrada para activar el proceso
    entrada = "Revisa el archivo base_datos.txt y actualiza los comentarios faltantes."
    
    try:
        # Usamos invoke para obtener el resultado final directamente
        result = agent.invoke({"messages": [("user", entrada)]})
        print("\n✅ Proceso completado.")
        print("Resumen del agente:", result["messages"][-1].content)
    except Exception as e:
        print(f"\n❌ Error de Cuota o Conexión: {e}")