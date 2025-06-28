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

# Função utilitária para chamada ao LLM Ollama

def consulta_llm(prompt: str, modelo: str = "llama3.2:1b", contexto: str = None) -> str:
    if contexto:
        prompt = f"{contexto}\n{prompt}"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": modelo,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[Erro ao consultar o LLM Ollama: {e}]"

class AgenteBase:
    def __init__(self, nome: str):
        self.nome = nome
    def log(self, mensagem: str):
        print(f"[{self.nome}] {mensagem}")

class InterpretaPerguntaAgente(AgenteBase):
    def __init__(self):
        super().__init__("InterpretaPergunta")
    def process(self, state: State) -> dict:
        self.log("Interpretando a pergunta com LLM Ollama...")
        prompt = f"Interprete a seguinte pergunta:\n{state['texto']}"
        interpretacao = consulta_llm(prompt)
        return {"texto": f"Pergunta interpretada: {interpretacao}"}

class BuscaContextoFilosoficoAgente(AgenteBase):
    def __init__(self):
        super().__init__("BuscaContextoFilosofico")
    def process(self, state: State) -> dict:
        self.log("Buscando contexto filosófico com LLM Ollama...")
        prompt = "Forneça um contexto filosófico para a resposta."
        contexto = consulta_llm(prompt, contexto=state['texto'])
        return {"texto": f"{state['texto']}\n{contexto}"}

class ElaboraRespostaAgente(AgenteBase):
    def __init__(self):
        super().__init__("ElaboraResposta")
    def process(self, state: State) -> dict:
        self.log("Elaborando resposta final com LLM Ollama...")
        prompt = "Responda de forma detalhada:"
        resposta_llm = consulta_llm(prompt, contexto=state['texto'])
        return {"texto": f"{state['texto']}\n{resposta_llm}"}

# Construção do grafo multiagente
if __name__ == "__main__":
    # Instanciando os agentes
    interpreta_agente = InterpretaPerguntaAgente()
    contexto_agente = BuscaContextoFilosoficoAgente()
    elabora_agente = ElaboraRespostaAgente()

    # Criação do grafo de estado
    builder = StateGraph(State)
    builder.add_node("interpreta", interpreta_agente.process)
    builder.add_node("contexto", contexto_agente.process)
    builder.add_node("elabora", elabora_agente.process)
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
# - Cada agente agora é uma classe com um método process.
# - O grafo executa o método process de cada agente.
# - Isso facilita a expansão e manutenção do sistema multiagente.
