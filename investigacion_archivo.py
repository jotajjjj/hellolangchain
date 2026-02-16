import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- HERRAMIENTAS (Tools) ---

@tool
def leer_archivo_memoria() -> str:
    """Lee el contenido de la base de datos local del asistente."""
    try:
        with open("base_datos.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "El archivo no existe todavía."

@tool
def escribir_en_memoria(contenido: str) -> str:
    """Guarda información nueva en la base de datos local. Úsalo para recordar cosas."""
    with open("base_datos.txt", "a") as f:
        f.write(f"\n{contenido}")
    return "Información guardada con éxito."

tools = [leer_archivo_memoria, escribir_en_memoria]

# --- CONFIGURACIÓN DEL CEREBRO (LLM) ---

# Usamos Gemini 2.5 Flash por su excelente seguimiento de instrucciones
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0
).bind_tools(tools)

# --- LÓGICA DEL GRAFO ---

def call_model(state: MessagesState):
    # El modelo recibe los mensajes y decide qué herramienta usar
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def router(state: MessagesState):
    # Mira el último mensaje: ¿Tiene llamadas a herramientas?
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- CONSTRUCCIÓN ---

workflow = StateGraph(MessagesState)

# Añadimos los nodos
workflow.add_node("assistant", call_model)
workflow.add_node("tools", ToolNode(tools))

# Definimos las conexiones
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", router)
workflow.add_edge("tools", "assistant") # Vuelve al asistente tras usar la herramienta

agent = workflow.compile()

# --- PRUEBA DEL AGENTE ---

if __name__ == "__main__":
    print("🤖 Agente de Investigación Activo")
    
    # Prueba 1: Preguntar algo que está en el archivo
    pregunta = "¿Cuál es el presupuesto del proyecto secreto?"
    print(f"\nUSER: {pregunta}")
    for chunk in agent.stream({"messages": [("user", pregunta)]}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    # Prueba 2: Pedirle que guarde algo nuevo
    orden = "Recuerda que el nuevo responsable del proyecto es Jose Jimenez."
    print(f"\nUSER: {orden}")
    for chunk in agent.stream({"messages": [("user", orden)]}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()