import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# --- 1. HERRAMIENTAS ---

@tool
def procesar_archivo_local() -> str:
    """Lee el archivo base_datos.txt y devuelve su contenido."""
    if not os.path.exists("base_datos.txt"):
        # Creamos un ejemplo si no existe para que no falle
        with open("base_datos.txt", "w", encoding="utf-8") as f:
            f.write("Pepe Romero, 38, Bombero\nMaria Garcia, 29, Ingeniera")
        return "Archivo creado con ejemplos."
    
    with open("base_datos.txt", "r", encoding="utf-8") as f:
        return f.read()

@tool
def guardar_mejoras(texto_final: str) -> str:
    """Sobreescribe el archivo con los nuevos comentarios."""
    with open("base_datos.txt", "w", encoding="utf-8") as f:
        f.write(texto_final)
    return "Base de datos actualizada localmente."

tools = [procesar_archivo_local, guardar_mejoras]

# --- 2. CONFIGURACIÓN DEL MODELO LOCAL ---

# Usamos Llama 3 a través de Ollama
llm = ChatOllama(
    model="qwen2.5:3b", 
    temperature=0  # Queremos que sea preciso, no creativo
).bind_tools(tools)

# --- 3. LÓGICA DEL AGENTE ---

def agente_editor(state: MessagesState):
    instrucciones = (
        "Eres un experto en RRHH. Tu objetivo es:\n"
        "1. Leer el archivo usando la herramienta.\n"
        "2. Identificar líneas que solo tienen 'Nombre, Edad, Profesión'.\n"
        "3. Añadir una 4ª columna con un comentario de MÁXIMO 10 PALABRAS sobre su profesión.\n"
        "4. El resultado final debe ser estrictamente: Nombre, Edad, Profesión, Comentario.\n"
        "5. Guardar el resultado final."
    )
    # Combinamos instrucciones y el historial de mensajes
    msg = [("system", instrucciones)] + state["messages"]
    response = llm.invoke(msg)
    return {"messages": [response]}

# --- 4. CONSTRUCCIÓN DEL GRAFO ---

def router(state: MessagesState):
    # Si el modelo decidió usar una herramienta, vamos al nodo de herramientas
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("asistente", agente_editor)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "asistente")
workflow.add_conditional_edges("asistente", router)
workflow.add_edge("tools", "asistente")

app = workflow.compile()

# --- 5. EJECUCIÓN ---

if __name__ == "__main__":
    print("🤖 Agente Local Ollama procesando datos...")
    input_usuario = {"messages": [("user", "Revisa mi base de datos y añade comentarios a las profesiones.")]}
    
    for evento in app.stream(input_usuario, stream_mode="values"):
        evento["messages"][-1].pretty_print()