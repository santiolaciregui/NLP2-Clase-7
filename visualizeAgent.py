# visualize_agent.py
"""
Genera imágenes PNG con la estructura interna de cada CVAgent y del router.
Requiere: pygraphviz y Graphviz instalados.
"""

from __future__ import annotations
import os, sys, re, traceback
from types import SimpleNamespace
from typing import Dict

# -----------------------------------------------------------------------------
# 1)  Rutas e importaciones
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)  # Para importar el módulo principal

try:
    from codigoNUEVO import CVAgent, _slug, MultiCVRouter
except ImportError as err:
    print("❌  No pude importar desde 'codigo_multiagente.py'.")
    print("    Asegúrate de que visualize_agent.py está junto a codigo_multiagente.py.")
    print("Detalle:", err)
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
# -----------------------------------------------------------------------------
# 2)  LLM dummy (solo para compilar el grafo)
# -----------------------------------------------------------------------------
if not OPENAI_API_KEY:
    print("⚠️  Define la variable de entorno OPENAI_API_KEY para inicializar el LLM.")
    sys.exit(1)

dummy_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# 3)  QA‑chain dummy que satisface la interfaz usada por CVAgent
# -----------------------------------------------------------------------------
class DummyQAch:
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, inp):
        # Devuelve un dict con la clave 'result', tal como espera CVAgent
        return {"result": f"[{self.tag}] → {inp['query']}"}

# -----------------------------------------------------------------------------
# 4)  Crear dos agentes de prueba
# -----------------------------------------------------------------------------
names = ["Santi Ejemplo"]
agents: Dict[str, CVAgent] = {}

for person in names:
    qa_dummy = DummyQAch(person)
    agents[person.lower()] = CVAgent(person, qa_dummy)

print("✅ Agentes dummy instanciados.")

# -----------------------------------------------------------------------------
# 5)  Compilar router
# -----------------------------------------------------------------------------
router = MultiCVRouter(agents, default_name=names[0].lower())

# -----------------------------------------------------------------------------
# 6)  Guardar PNG de cada grafo
# -----------------------------------------------------------------------------
def save_graph(component, filename: str):
    try:
        png = component.get_graph().draw_png()
        with open(filename, "wb") as fh:
            fh.write(png)
        print(f"📄 Guardado → {filename}")
    except ImportError:
        print("⚠️  Instala pygraphviz para exportar PNG:  pip install pygraphviz")
        sys.exit(1)
    except FileNotFoundError:
        print("⚠️  Graphviz (dot) no está en tu PATH. Descárgalo en https://graphviz.org/download/")
        sys.exit(1)
    except Exception as e:
        print("❌  Error al generar el PNG:", e)
        traceback.print_exc()

os.makedirs("graphs", exist_ok=True)

# a) Grafos internos de cada agente
for person, ag in agents.items():
    fname = f"graphs/agent_{_slug(person)}.png"
    save_graph(ag.graph, fname)

# b) Grafo del router (solo nodos de routing, no de LLM)
fname_router = "graphs/router_flow.png"
save_graph(router.name_pattern, fname_router)  # name_pattern es un re.Pattern, no un grafo

print("\n¡Listo! Revisa la carpeta 'graphs/'.")

