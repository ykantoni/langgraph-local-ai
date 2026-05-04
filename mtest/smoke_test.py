import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from local_agent import chat_agent, cli, vectorstore
from local_agent.config import Settings


class SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            log_level="INFO",
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="granite3.3",
            st_embed_model="sentence-transformers/static-retrieval-mrl-en-v1",
            st_encode_batch=32,
            st_device="cpu",
            docs_dir="C:/tmp/docs",
            faiss_index_dir="C:/tmp/faiss_store",
            faiss_index_name="index",
            embed_max_chars=1800,
            embed_batch_size=32,
            embed_threads=8,
            pdf_max_pages=0,
            pdf_extractor="auto",
            pdf_cache_dir="C:/tmp/pdf_cache",
            force_rebuild_index=False,
        )

    def test_cli_main_wires_components_and_exits_without_running_agent(self) -> None:
        fake_embeddings = object()
        fake_vectorstore = object()
        fake_agent = MagicMock()

        with (
            patch("local_agent.cli.load_settings", return_value=self.settings) as load_settings_mock,
            patch("local_agent.cli.setup_logging") as setup_logging_mock,
            patch("local_agent.cli.SentenceTransformerEmbeddings", return_value=fake_embeddings) as embeddings_cls,
            patch("local_agent.cli.get_or_create_vectorstore", return_value=fake_vectorstore) as get_vectorstore_mock,
            patch("local_agent.cli.create_chat_agent", return_value=fake_agent) as create_chat_agent_mock,
            patch("builtins.input", side_effect=KeyboardInterrupt),
            patch("builtins.print") as print_mock,
        ):
            cli.main()

        load_settings_mock.assert_called_once_with()
        setup_logging_mock.assert_called_once_with(self.settings.log_level)
        embeddings_cls.assert_called_once_with(
            model_name=self.settings.st_embed_model,
            encode_batch_size=self.settings.st_encode_batch,
            device=self.settings.st_device,
        )
        get_vectorstore_mock.assert_called_once_with(self.settings, fake_embeddings)
        create_chat_agent_mock.assert_called_once_with(fake_vectorstore, self.settings)
        fake_agent.run.assert_not_called()
        print_mock.assert_called_once_with("\nExiting.")

    def test_create_chat_agent_wires_tool_and_llm(self) -> None:
        vectorstore_mock = MagicMock()
        vectorstore_mock.similarity_search.return_value = [
            Document(page_content="first result", metadata={}),
            Document(page_content="second result", metadata={}),
        ]
        llm_instance = object()
        agent_instance = object()

        with (
            patch("local_agent.chat_agent.ChatOllama", return_value=llm_instance) as chat_ollama_mock,
            patch("local_agent.chat_agent.initialize_agent", return_value=agent_instance) as initialize_agent_mock,
        ):
            created_agent = chat_agent.create_chat_agent(vectorstore_mock, self.settings)

        self.assertIs(created_agent, agent_instance)
        chat_ollama_mock.assert_called_once_with(
            model=self.settings.ollama_chat_model,
            temperature=0,
            base_url=self.settings.ollama_base_url,
        )
        initialize_agent_mock.assert_called_once()

        kwargs = initialize_agent_mock.call_args.kwargs
        self.assertEqual(kwargs["llm"], llm_instance)
        self.assertEqual(kwargs["agent"], chat_agent.AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        self.assertTrue(kwargs["verbose"])
        self.assertEqual(len(kwargs["tools"]), 1)

        search_tool = kwargs["tools"][0]
        tool_result = search_tool.func("test query")
        vectorstore_mock.similarity_search.assert_called_once_with("test query", k=3)
        self.assertEqual(tool_result, "first result\n\nsecond result")

    def test_get_or_create_vectorstore_uses_saved_index_when_docs_unchanged(self) -> None:
        embeddings = MagicMock()
        loaded_vectorstore = object()

        with (
            patch("local_agent.vectorstore.os.path.exists", return_value=True),
            patch("local_agent.vectorstore.docs_changed", return_value=False),
            patch("local_agent.vectorstore.FAISS.load_local", return_value=loaded_vectorstore) as load_local_mock,
            patch("local_agent.vectorstore.load_documents") as load_documents_mock,
            patch("local_agent.vectorstore.build_faiss_batched") as build_faiss_mock,
        ):
            result = vectorstore.get_or_create_vectorstore(self.settings, embeddings)

        self.assertIs(result, loaded_vectorstore)
        load_local_mock.assert_called_once_with(
            folder_path=self.settings.faiss_index_dir,
            embeddings=embeddings,
            index_name=self.settings.faiss_index_name,
            allow_dangerous_deserialization=True,
        )
        load_documents_mock.assert_not_called()
        build_faiss_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

