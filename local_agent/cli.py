import logging

from local_agent.chat_agent import create_chat_agent
from local_agent.config import load_settings
from local_agent.vectorstore import SentenceTransformerEmbeddings, get_or_create_vectorstore


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    embeddings = SentenceTransformerEmbeddings(
        model_name=settings.st_embed_model,
        encode_batch_size=settings.st_encode_batch,
        device=settings.st_device,
    )
    vectorstore = get_or_create_vectorstore(settings, embeddings)
    agent = create_chat_agent(vectorstore, settings)

    while True:
        try:
            query = input("Ask: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        response = agent.run(query)
        print("\nAnswer:", response)

