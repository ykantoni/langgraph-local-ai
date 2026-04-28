import os
from pathlib import Path

from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_documents(docs_dir: str = "./docs"):
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    docs = []
    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            loader = TextLoader(str(docs_path / file))
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .txt files found in {docs_path.resolve()}")
    return docs


# 1. Load local documents
docs = load_documents("/docs")

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create vector index (FAISS)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)


# 4. Create a search function
def search_docs(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([r.page_content for r in results])


# 5. Wrap as a tool
search_tool = Tool(
    name="LocalDocumentSearch",
    func=search_docs,
    description="Searches local documents for relevant information",
)

# 6. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 7. Agent (decides when to use tool)
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 8. Ask questions
while True:
    try:
        query = input("Ask: ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break
    response = agent.run(query)
    print("\nAnswer:", response)