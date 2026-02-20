from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from memora.store import Store
from memora.retriever import HybridRetriever
from memora.config import settings
from dotenv import load_dotenv

load_dotenv()

SYSTEM = """You are Memora, a helpful memory assistant. Answer based on the provided context.
If the context doesn't contain relevant info, say so. Be concise and accurate.
Cite the source file when referencing information.

Context:
{context}"""

_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])


class RAG:
    def __init__(self):
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to .env or environment.")
        self.store = Store()
        self.retriever = HybridRetriever(self.store)
        self.llm = ChatOpenAI(
            model=settings.openrouter_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            default_headers={"HTTP-Referer": "https://github.com/memora", "X-Title": "Memora"},
        )
        self.chain = _prompt | self.llm | StrOutputParser()

    def _build_context(self, question: str) -> str:
        docs = self.retriever.retrieve(question)
        if not docs:
            return "No documents found."
        return "\n---\n".join(f"[{d['source']}]\n{d['text']}" for d in docs)

    def ask(self, question: str) -> str:
        ctx = self._build_context(question)
        return self.chain.invoke({"context": ctx, "question": question})

    def ask_stream(self, question: str):
        ctx = self._build_context(question)
        for chunk in self.chain.stream({"context": ctx, "question": question}):
            yield chunk
