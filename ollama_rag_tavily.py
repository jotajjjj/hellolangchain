import os
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# --- IMPORTACIÓN BASADA EN TU LIBRERÍA ---
from langchain_tavily import TavilySearch 

load_dotenv()

# --- HERRAMIENTAS ---

@tool
def crear_informe_pdf(contenido: str):
    """Genera un archivo PDF con la investigación."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    texto_seguro = contenido.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=texto_seguro)
    pdf.output("Informe_Investigacion.pdf")
    return "✅ PDF creado."

# Usamos el nombre que viste en el __all__
search_tool = TavilySearch(
    max_results=1, 
    description="ÚSALO PARA BUSCAR NOTICIAS REALES EN INTERNET. Es obligatorio usarlo antes de hacer el PDF."
)
@tool
def leer_archivo() -> str:
    """Lee el archivo base_datos.txt."""
    with open("base_datos.txt", "r", encoding="utf-8") as f:
        return f.read()

tools = [leer_archivo, crear_informe_pdf, search_tool]

# --- MODELO ---
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

# --- NODO DEL AGENTE ---
def agente_investigador(state: MessagesState):
    prompt = (
        "Eres un robot de procesamiento de datos. NO respondas con texto, SOLO usa herramientas.\n"
        "TU FLUJO ES:\n"
        "1. Leer el archivo.\n"
        "2. Por cada profesión, llamar a 'tavily_search'.\n"
        "3. Una vez tengas la info de TODAS las profesiones, llama a 'crear_informe_pdf'.\n"
        "NO te detengas a explicar lo que encuentras. Pasa directamente de una herramienta a otra."
    )
    # Importante: mantenemos el historial para que no olvide lo que buscó
    res = llm.invoke([("system", prompt)] + state["messages"])
    return {"messages": [res]}
# --- GRAFO ---
def router(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("asistente", agente_investigador)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "asistente")
workflow.add_conditional_edges("asistente", router)
workflow.add_edge("tools", "asistente")
app = workflow.compile()

import time

if __name__ == "__main__":
    print("✨ ¡Todo listo! Lanzando investigación con TavilySearch...")
    inicio = time.time()
    
    # Usamos .stream para ver el proceso en tiempo real
    entrada = {"messages": [("user", "Investiga las profesiones de mi archivo y genera el PDF.")]}
    
    try:
        # Esto imprimirá cada paso que dé el agente
        for paso in app.stream(entrada, stream_mode="values"):
            ultimo_mensaje = paso["messages"][-1]
            ultimo_mensaje.pretty_print()
            
        print(f"\n✅ Proceso terminado en {round(time.time() - inicio, 2)} segundos.")
    except Exception as e:
        print(f"❌ Ocurrió un error durante la ejecución: {e}")