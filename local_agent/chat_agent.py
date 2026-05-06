from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from local_agent.config import Settings


def create_chat_agent(vectorstore, settings: Settings):
    def search_docs(query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([r.page_content for r in results])

    search_tool = Tool(
        name="LocalDocumentSearch",
        func=search_docs,
        description="Searches local documents for relevant information",
    )

    llm = ChatOllama(
        model=settings.ollama_chat_model,
        temperature=0,
        base_url=settings.ollama_base_url,
    )

    return initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

