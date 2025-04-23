from __future__ import annotations
import os, re, time, tempfile, unicodedata, operator, traceback, random, string
from typing import Dict, List, Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

if not (PINECONE_API_KEY and GROQ_API_KEY and OPENAI_API_KEY):
    st.error("Faltan variables de entorno necesarias.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_models():
    chat_llama3 = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    return chat_llama3, embeddings

def _slug(txt: str, maxlen: int = 63) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-zA-Z0-9_-]+", "_", txt.lower()).strip("_")
    return txt[:maxlen] or ''.join(random.choices(string.ascii_lowercase, k=8))

def read_pdf(uploaded) -> List:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue()); path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
        st.success(f"✔️  {uploaded.name}: {len(docs)} páginas")
        return docs
    finally:
        os.remove(path)

def chunkify(docs, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def build_retriever(index_name, namespace, embed_model, chunks):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=3072, metric="cosine", spec=spec)
        time.sleep(1)

    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    vs = PineconeVectorStore(
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    return vs.as_retriever(search_kwargs={"k": 4})

def make_cv_tool(person: str, retriever) -> Tool:
    safe = _slug(person)
    return Tool(
        name=f"lookup_{safe}",
        description=f"Busca información únicamente en el CV de {person}. Argumento: 'query' (str).",
        func=lambda q, r=retriever: r.get_relevant_documents(q)
    )

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class MultiCVAgent:
    def __init__(self, persons_tools: Dict[str, Tool], system_prompt: str):
        tools = list(persons_tools.values())
        self.tool_lookup = {t.name: t for t in tools}

        model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0
        ).bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self._call_llm(model, system_prompt))
        graph.add_node("act", self._take_action)
        graph.add_conditional_edges("llm", self._has_action, {True: "act", False: END})
        graph.add_edge("act", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    @staticmethod
    def _has_action(state: AgentState) -> bool:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        return isinstance(last, AIMessage) and bool(tool_calls)

    def _call_llm(self, model, sys_prompt):
        sys_msg = SystemMessage(content=sys_prompt)
        def _inner(state: AgentState):
            msgs = [sys_msg] + state["messages"]
            resp = model.invoke(msgs)
            return {"messages": [resp]}
        return _inner

    def _take_action(self, state: AgentState):
        ai_msg = state["messages"][-1]
        out: List[ToolMessage] = []
        for tc in ai_msg.tool_calls:
            tool = self.tool_lookup.get(tc["name"])
            try:
                docs = tool.invoke(tc["args"]["query"])
                snippets = "\n".join(d.page_content[:400] for d in docs) or "Sin resultados."
                out.append(ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=snippets))
            except Exception as e:
                out.append(ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=f"ERROR herramienta: {e}"))
        return {"messages": out}

    def __call__(self, question: str) -> str:
        result = self.graph.invoke({"messages": [HumanMessage(content=question)]})
        return result["messages"][-1].content

st.set_page_config("Comparador de CVs", layout="wide")
st.title("📄🤖 Comparador de CVs con LangGraph (Groq + Pinecone)")

with st.sidebar:
    st.markdown("## Parámetros")
    chunk_size    = st.slider("Chunk size", 500, 4000, 2000, 100)
    chunk_overlap = st.slider("Overlap", 0, 500, 200, 10)
    pinecone_idx  = st.text_input("Pinecone index", value="cv-index")
    st.markdown("---")
    st.markdown("Sube dos o más CVs en PDF y asigna un nombre claro a cada uno.")

uploads = st.file_uploader("CVs en PDF", type="pdf", accept_multiple_files=True)

if uploads:
    st.header("Nombres para cada CV")
    names: List[str] = []
    for i, f in enumerate(uploads, 1):
        default = os.path.splitext(f.name)[0]
        names.append(st.text_input(f"Nombre CV #{i}", value=default))

    if st.button("🚀 Procesar CVs"):
        st.session_state.clear()
        chat_llm, embed_model = get_models()

        persons_tools: Dict[str, Tool] = {}
        for upl, nm in zip(uploads, names):
            namespace = f"cv_{_slug(nm)}"
            with st.spinner(f"Procesando {nm}…"):
                docs   = read_pdf(upl)
                chunks = chunkify(docs, chunk_size, chunk_overlap)
                retr   = build_retriever(pinecone_idx, namespace, embed_model, chunks)
                persons_tools[nm] = make_cv_tool(nm, retr)

        tool_list = "\n".join(f"- {n}: `{t.name}`" for n, t in persons_tools.items())
        sys_prompt = (
            "Eres un analista de RRHH experto. Usa las herramientas para responder "
            "preguntas sobre los CVs.\n"
            "Reglas:\n"
            "1. Si la pregunta menciona claramente solo uno de los nombres, usa "
            "exclusivamente la herramienta de ese CV.\n"
            "2. Si la pregunta menciona más de un nombre o ninguno, usa todas las "
            "herramientas relevantes y fusiona la respuesta citando a cada persona.\n"
            "3. Nunca inventes información que no esté en los CVs.\n\n"
            f"### CVs y herramientas\n{tool_list}"
        )

        st.session_state["agent"] = MultiCVAgent(persons_tools, sys_prompt)
        st.success("✅ Agente configurado. ¡Ya puedes preguntar!")

if "agent" in st.session_state:
    query = st.chat_input("Escribe tu pregunta…")
    if query:
        with st.chat_message("user"):      st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Consultando…"):
                try:
                    answer = st.session_state["agent"](query)
                except Exception:
                    answer = f"⚠️ Error interno:\n```\n{traceback.format_exc()}\n```"
            st.markdown(answer)
