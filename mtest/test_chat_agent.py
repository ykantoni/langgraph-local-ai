import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from local_agent import chat_agent, cli, vectorstore
from local_agent.config import Settings

# show top five most frequent Replication Server errors found in local documents
# what are potential issues when compiling ganymed library

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
            faiss_nlist=256,
            faiss_pq_m=64,
            faiss_pq_nbits=8,
            faiss_nprobe=16,
            embed_max_chars=1800,
            embed_batch_size=32,
            embed_threads=8,
            pdf_max_pages=0,
            pdf_extractor="auto",
            pdf_cache_dir="C:/tmp/pdf_cache",
            force_rebuild_index=False,
            mcp_enabled=False,
            mcp_servers_json="",
            mcp_tool_search_name="search_docs",
        )

    def test_cli_main_wires_components_and_exits_without_running_agent(self) -> None:
        fake_agent = MagicMock()

        with (
            patch("local_agent.cli.load_chat_agent", return_value=(fake_agent, self.settings)) as load_chat_agent_mock,
            patch("builtins.input", side_effect=KeyboardInterrupt),
            patch("builtins.print") as print_mock,
        ):
            cli.main()

        load_chat_agent_mock.assert_called_once_with()
        fake_agent.run.assert_not_called()
        print_mock.assert_called_once_with("\nExiting.")

    def test_create_chat_agent_wires_tool_and_llm(self) -> None:
        vectorstore_mock = MagicMock()
        vectorstore_mock.similarity_search.return_value = [
            Document(page_content="first result", metadata={}),
            Document(page_content="second result", metadata={}),
        ]
        llm_instance = MagicMock()

        captured_tool = {}

        class FakeTool:
            def __init__(self, name: str, func, description: str):
                self.name = name
                self.func = func
                self.description = description
                captured_tool["tool"] = self

        plan = (
            "1) step one\n###\n"
            "2) step two\n###\n"
            "3) step three\n###\n"
        )
        # planner, three executor steps, synthesizer, critic (PASS)
        llm_instance.invoke.side_effect = [
            AIMessage(content=plan),
            AIMessage(content="(executor 1)"),
            AIMessage(content="(executor 2)"),
            AIMessage(content="(executor 3)"),
            AIMessage(content="Final answer"),
            AIMessage(content="PASS"),
        ]

        with (
            patch("local_agent.chat_agent.ChatOllama", return_value=llm_instance) as chat_ollama_mock,
            patch("local_agent.chat_agent.Tool", FakeTool),
        ):
            created_agent = chat_agent.create_chat_agent(vectorstore_mock, self.settings)

        self.assertTrue(hasattr(created_agent, "graph"))
        chat_ollama_mock.assert_called_once_with(
            model=self.settings.ollama_chat_model,
            temperature=0,
            base_url=self.settings.ollama_base_url,
            streaming=True,
        )

        # Ensure tool function is correctly wired to the vectorstore.
        search_tool = captured_tool["tool"]
        tool_result = search_tool.func("test query")
        vectorstore_mock.similarity_search.assert_called_once_with("test query", k=3)
        self.assertEqual(tool_result, "first result\n\nsecond result")

        # Ensure the resulting graph is invokable and returns agent state.
        result = created_agent.invoke({"input": "hello"})
        self.assertEqual(result.get("question"), "hello")
        self.assertEqual(result.get("final_answer"), "Final answer")

    def test_create_chat_agent_tool_calling_path_invokes_mcp_tool(self) -> None:
        vectorstore_mock = MagicMock()
        vectorstore_mock.similarity_search.return_value = [
            Document(page_content="fallback only", metadata={}),
        ]

        mcp_tool = MagicMock()
        mcp_tool.name = "search_docs"
        mcp_tool.invoke.return_value = "MCP TOOL RESULT"

        llm_instance = MagicMock()
        bound_llm = MagicMock()
        llm_instance.bind_tools.return_value = bound_llm

        plan = (
            "1) step one\n###\n"
            "2) step two\n###\n"
            "3) step three\n###\n"
        )
        # planner, bound retrieve, 3 executors, synthesizer, critic (PASS)
        llm_instance.invoke.side_effect = [
            AIMessage(content=plan),
            AIMessage(content="(executor 1)"),
            AIMessage(content="(executor 2)"),
            AIMessage(content="(executor 3)"),
            AIMessage(content="Final answer"),
            AIMessage(content="PASS"),
        ]
        bound_llm.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "search_docs", "args": {"query": "hello"}},
            ],
        )

        with patch("local_agent.chat_agent.ChatOllama", return_value=llm_instance):
            created_agent = chat_agent.create_chat_agent(
                vectorstore_mock,
                self.settings,
                tools=[mcp_tool],
                tools_supported=True,
            )

        result = created_agent.invoke({"input": "hello"})

        llm_instance.bind_tools.assert_called_once_with([mcp_tool])
        bound_llm.invoke.assert_called_once()
        mcp_tool.invoke.assert_called_once()
        # MCP tool output should land in intermediate_results (in addition to executor outputs).
        self.assertIn("MCP TOOL RESULT", result.get("intermediate_results"))
        # Fallback FAISS path must NOT be hit when the MCP tool returns content.
        vectorstore_mock.similarity_search.assert_not_called()
        self.assertEqual(result.get("final_answer"), "Final answer")

    def test_create_chat_agent_falls_back_when_bind_tools_unsupported(self) -> None:
        """If bind_tools raises (model doesn't actually support tools), use direct retrieval."""
        vectorstore_mock = MagicMock()
        vectorstore_mock.similarity_search.return_value = [
            Document(page_content="local search result", metadata={}),
        ]

        mcp_tool = MagicMock()
        mcp_tool.name = "search_docs"

        llm_instance = MagicMock()
        llm_instance.bind_tools.side_effect = NotImplementedError("no tools")

        plan = (
            "1) step one\n###\n"
            "2) step two\n###\n"
            "3) step three\n###\n"
        )
        llm_instance.invoke.side_effect = [
            AIMessage(content=plan),
            AIMessage(content="(executor 1)"),
            AIMessage(content="(executor 2)"),
            AIMessage(content="(executor 3)"),
            AIMessage(content="Final answer"),
            AIMessage(content="PASS"),
        ]

        with patch("local_agent.chat_agent.ChatOllama", return_value=llm_instance):
            created_agent = chat_agent.create_chat_agent(
                vectorstore_mock,
                self.settings,
                tools=[mcp_tool],
                tools_supported=True,
            )

        result = created_agent.invoke({"input": "hello"})

        mcp_tool.invoke.assert_not_called()
        vectorstore_mock.similarity_search.assert_called_once_with("hello", k=3)
        self.assertIn("local search result", result.get("intermediate_results"))
        self.assertEqual(result.get("final_answer"), "Final answer")

    def test_get_or_create_vectorstore_uses_saved_index_when_docs_unchanged(self) -> None:
        embeddings = MagicMock()
        loaded_vectorstore = object()

        with (
            patch("local_agent.vectorstore.os.path.exists", return_value=True),
            patch("local_agent.vectorstore.docs_changed", return_value=False),
            patch("local_agent.vectorstore.FAISS.load_local", return_value=loaded_vectorstore) as load_local_mock,
            patch("local_agent.vectorstore.load_documents") as load_documents_mock,
            patch("local_agent.vectorstore.build_faiss_ivfpq_batched") as build_faiss_mock,
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

