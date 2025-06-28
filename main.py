import langgraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
import requests

# Exemplo de sistema Chain-of-Thought (CoT) multiagente para responder:
# "Qual é o sentido da vida, e por que é 42?"
# Cada agente é um nó, e as conexões são as arestas do grafo.

# Definição do estado compartilhado entre os nós
class State(TypedDict):
    texto: str

# Funções dos nós adaptadas para o novo padrão

def interpreta_pergunta(state: State) -> dict:
    print("[InterpretaPergunta] Interpretando a pergunta...")
    return {"texto": f"Pergunta interpretada: {state['texto']}"}

def busca_contexto_filosofico(state: State) -> dict:
    print("[BuscaContextoFilosofico] Buscando contexto filosófico...")
    contexto = "Na obra 'O Guia do Mochileiro das Galáxias', 42 é a resposta para a pergunta fundamental da vida."
    return {"texto": f"{state['texto']}\n{contexto}"}

def elabora_resposta(state: State) -> dict:
    print("[ElaboraResposta] Elaborando resposta final com LLM Ollama...")
    prompt = f"{state['texto']}\nResponda de forma detalhada:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",  # Altere para o modelo disponível no seu Ollama
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        resposta_llm = data.get("response", "").strip()
    except Exception as e:
        resposta_llm = f"[Erro ao consultar o LLM Ollama: {e}]"
    return {"texto": f"{state['texto']}\n{resposta_llm}"}

# Construção do grafo multiagente
if __name__ == "__main__":
    # Criação do grafo de estado
    builder = StateGraph(State)
    builder.add_node("interpreta", interpreta_pergunta)
    builder.add_node("contexto", busca_contexto_filosofico)
    builder.add_node("elabora", elabora_resposta)
    builder.add_edge("interpreta", "contexto")
    builder.add_edge("contexto", "elabora")
    builder.set_entry_point("interpreta")
    builder.set_finish_point("elabora")
    graph = builder.compile()

    # Execução do fluxo
    entrada = {"texto": "Qual é o sentido da vida, e por que é 42?"}
    resultado = graph.invoke(entrada)
    print("\n[Resultado Final]\n", resultado["texto"])

# Comentários explicativos:
# - Cada nó é um agente com uma função específica.
# - O grafo define a ordem de execução dos agentes.
# - O fluxo pode ser expandido com mais agentes ou lógica condicional.
