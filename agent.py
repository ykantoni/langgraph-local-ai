import os
from pathlib import Path

from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_documents(docs_dir: str):
    """Load documents with fallback encoding handling."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    docs = []
    # Try encodings in order: UTF-8, Windows-1252 (common on Windows), Latin-1, ASCII
    encodings_to_try = ["utf-8", "cp1252", "latin-1", "ascii"]
    
    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            file_path = str(docs_path / file)
            successfully_loaded = False
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, encoding=encoding) as f:
                        text = f.read()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path}
                    ))
                    successfully_loaded = True
                    print(f"✓ Loaded {file} with {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if not successfully_loaded:
                raise RuntimeError(f"Could not load {file} with any encoding: {encodings_to_try}")

    if not docs:
        raise ValueError(f"No .txt files found in {docs_path.resolve()}")
    return docs


def configure_openai_api_key(api_key: str | None = None, override: bool = False) -> None:
    """Set OPENAI_API_KEY programmatically with safe defaults."""
    existing_key = os.environ.get("OPENAI_API_KEY")
    if existing_key and not override and api_key is None:
        return

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY_VALUE")

    if api_key is None:
        raise ValueError(
            "OPENAI_API_KEY is not set. Set OPENAI_API_KEY in your environment "
            "or pass a key to configure_openai_api_key()."
        )

    if not isinstance(api_key, str):
        raise TypeError("OPENAI_API_KEY must be a string.")

    api_key = api_key.strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY cannot be empty.")

    os.environ["OPENAI_API_KEY"] = api_key



# 1. Load local documents
docs = load_documents("./mtest")

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create vector index (FAISS)
# Ensure OPENAI_API_KEY exists before creating OpenAI-backed clients.
configure_openai_api_key("<REDACTED_OPENAI_API_KEY>")
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