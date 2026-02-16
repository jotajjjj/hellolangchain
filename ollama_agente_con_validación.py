import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# --- 1. HERRAMIENTAS (TOOLS) ---
@tool
def leer_archivo() -> str:
    """Lee el contenido de base_datos.txt."""
    if not os.path.exists("base_datos.txt"):
        return "Error: El archivo no existe."
    with open("base_datos.txt", "r", encoding="utf-8") as f:
        return f.read()

@tool
def escribir_archivo(contenido: str) -> str:
    """Sobreescribe el archivo con los datos validados y comentados."""
    with open("base_datos.txt", "w", encoding="utf-8") as f:
        f.write(contenido)
    return "Archivo actualizado y validado."

tools = [leer_archivo, escribir_archivo]

# --- 2. EL MODELO (LLM) ---
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

# --- 3. LÓGICA DEL AGENTE ---
def validador_y_editor(state: MessagesState):
    prompt_sistema = (
        "Eres un supervisor de datos. Tu flujo de trabajo es:\n"
        "1. Lee el archivo.\n"
        "2. Revisa línea por línea: Nombre, Edad, Profesión.\n"
        "3. VALIDACIÓN: Si la Edad no es un número, NO guardes esa línea y genera un aviso de error, poniendo INVALIDO.\n"
        "4. Si es válida y no tiene comentario, añade uno (máx 10 palabras).\n"
        "5. Si todo está bien, guarda el archivo final: Nombre, Edad, Profesión, Comentario."
    )
    # LangChain gestiona los mensajes como una lista de objetos
    mensajes = [("system", prompt_sistema)] + state["messages"]
    respuesta = llm.invoke(mensajes)
    return {"messages": [respuesta]}

# --- 4. GRAFO (EL MOTOR) ---
def direccionador(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("agente", validador_y_editor)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agente")
workflow.add_conditional_edges("agente", direccionador)
workflow.add_edge("tools", "agente")

app = workflow.compile()

if __name__ == "__main__":
    # Prueba con un dato erróneo en el archivo antes de correrlo:
    # Ej: "Carlos, VEINTE, Pintor"
    app.invoke({"messages": [("user", "Valida y actualiza mi base de datos.")]})