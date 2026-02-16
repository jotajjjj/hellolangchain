import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
# CAMBIO AQUÍ: Importación desde la ruta actualizada de langgraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def obtener_clima(ciudad: str) -> str:
    """Consulta el clima actual de una ciudad."""
    if "madrid" in ciudad.lower():
        return "Soleado, 25°C"
    return "Nublado, 18°C"

tools = [obtener_clima]

# Usamos el modelo que confirmamos que funciona
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0
)

# Creamos el agente (esta función ahora vive en langgraph.prebuilt)
app = create_react_agent(llm, tools)

def ejecutar_agente(pregunta: str):
    print(f"\n--- Usuario: {pregunta} ---")
    inputs = {"messages": [("user", pregunta)]}
    
    # Ejecutamos y obtenemos el resultado final
    result = app.invoke(inputs)
    
    # El último mensaje en la lista es la respuesta del agente
    respuesta_final = result["messages"][-1].content
    print(f"Agente: {respuesta_final}")

if __name__ == "__main__":
    ejecutar_agente("¿Qué tiempo hace en Madrid?")