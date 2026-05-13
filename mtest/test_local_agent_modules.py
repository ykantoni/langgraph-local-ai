"""Unit tests for local_agent server, ingest, runtime, cli, and vectorstore modules."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from starlette.testclient import TestClient

from local_agent import cli, ingest, runtime, server as server_mod, vectorstore
from local_agent.config import Settings
from local_agent.server import ChatMessage, app


def _minimal_settings(**overrides) -> Settings:
    base = dict(
        log_level="INFO",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="granite3.3",
        st_embed_model="sentence-transformers/static-retrieval-mrl-en-v1",
        st_encode_batch=32,
        st_device="cpu",
        docs_dir="/tmp/docs",
        faiss_index_dir="/tmp/faiss",
        faiss_index_name="index",
        faiss_nlist=256,
        faiss_pq_m=64,
        faiss_pq_nbits=8,
        faiss_nprobe=16,
        embed_max_chars=1800,
        embed_batch_size=32,
        embed_threads=2,
        pdf_max_pages=0,
        pdf_extractor="auto",
        pdf_cache_dir="/tmp/pdf_cache",
        force_rebuild_index=False,
        mcp_enabled=False,
        mcp_servers_json="",
        mcp_tool_search_name="search_docs",
    )
    base.update(overrides)
    return Settings(**base)


class TestServerHelpers(unittest.TestCase):
    def test_build_input_no_history(self) -> None:
        self.assertEqual(server_mod._build_input("hello", None), "hello")
        self.assertEqual(server_mod._build_input("hello", []), "hello")

    def test_build_input_with_history(self) -> None:
        hist = [
            ChatMessage(role="user", content="u1"),
            ChatMessage(role="assistant", content="a1"),
        ]
        out = server_mod._build_input("u2", hist)
        self.assertIn("User: u1", out)
        self.assertIn("Assistant: a1", out)
        self.assertTrue(out.endswith("User: u2"))

    def test_new_session_id_is_hex(self) -> None:
        sid = server_mod._new_session_id()
        self.assertEqual(len(sid), 32)
        int(sid, 16)  # raises if not hex


class TestServerHTTP(unittest.TestCase):
    def tearDown(self) -> None:
        server_mod._sessions.clear()

    def _client_with_agent(self, agent: MagicMock):
        return patch(
            "local_agent.server.load_chat_agent",
            return_value=(agent, MagicMock()),
        )

    def test_health(self) -> None:
        agent = MagicMock()
        with self._client_with_agent(agent), TestClient(app) as client:
            r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_session_new_returns_distinct_ids_and_empty_history(self) -> None:
        agent = MagicMock()
        with self._client_with_agent(agent), TestClient(app) as client:
            r = client.post("/session/new")
            self.assertEqual(r.status_code, 200)
            sid = r.json()["session_id"]
            self.assertEqual(len(sid), 32)
            server_mod._append_session_message(sid, ChatMessage(role="user", content="x"))
            r2 = client.post("/session/new")
            sid2 = r2.json()["session_id"]
        self.assertNotEqual(sid, sid2)
        self.assertEqual(server_mod._get_session_history(sid2), [])
        self.assertEqual(len(server_mod._get_session_history(sid)), 1)

    def test_chat_stream_503_when_agent_unset(self) -> None:
        agent = MagicMock()
        with self._client_with_agent(agent), TestClient(app) as client:
            server_mod._agent = None
            r = client.post("/chat/stream", json={"message": "hi"})
        self.assertEqual(r.status_code, 503)

    def test_chat_stream_emits_start_done(self) -> None:
        agent = MagicMock()

        def _invoke(_payload, config=None):
            for cb in (config or {}).get("callbacks") or []:
                cb.on_llm_new_token("Hi.")
            return {}

        agent.invoke.side_effect = _invoke

        with self._client_with_agent(agent), TestClient(app) as client:
            with client.stream("POST", "/chat/stream", json={"message": "hello"}) as r:
                self.assertEqual(r.status_code, 200)
                body = "".join(r.iter_text())

        self.assertIn('"type": "start"', body)
        self.assertIn('"type": "done"', body)
        agent.invoke.assert_called()
        args, _kw = agent.invoke.call_args
        self.assertEqual(args[0], {"input": "hello"})

    def test_chat_stream_with_session_passes_history(self) -> None:
        agent = MagicMock()

        def _invoke(_payload, config=None):
            for cb in (config or {}).get("callbacks") or []:
                cb.on_llm_new_token("Ok.")
            return {}

        agent.invoke.side_effect = _invoke

        with self._client_with_agent(agent), TestClient(app) as client:
            s = client.post("/session/new").json()["session_id"]
            client.post("/chat/stream", json={"message": "first", "session_id": s})
            with client.stream(
                "POST",
                "/chat/stream",
                json={"message": "second", "session_id": s},
            ) as r:
                self.assertEqual(r.status_code, 200)
                body = "".join(r.iter_text())

        self.assertIn('"type": "done"', body)
        last_input = agent.invoke.call_args_list[-1][0][0]["input"]
        self.assertIn("User: first", last_input)
        self.assertIn("Assistant:", last_input)
        self.assertTrue(last_input.endswith("User: second"))

    def test_chat_request_rejects_empty_message(self) -> None:
        agent = MagicMock()
        with self._client_with_agent(agent), TestClient(app) as client:
            r = client.post("/chat/stream", json={"message": ""})
        self.assertEqual(r.status_code, 422)


class TestIngest(unittest.TestCase):
    def test_load_txt_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("hello é", encoding="utf-8")
            settings = _minimal_settings(docs_dir=d)
            docs = ingest.load_documents(settings)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "hello é")
        self.assertEqual(docs[0].metadata.get("type"), "txt")

    def test_load_documents_missing_dir(self) -> None:
        settings = _minimal_settings(docs_dir="/nonexistent/dir/ingest_test")
        with self.assertRaises(FileNotFoundError):
            ingest.load_documents(settings)

    def test_load_documents_no_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "readme.md").write_text("x", encoding="utf-8")
            settings = _minimal_settings(docs_dir=d)
            with self.assertRaises(ValueError) as ctx:
                ingest.load_documents(settings)
        self.assertIn("No supported files", str(ctx.exception))

    def test_collect_docs_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            t1 = Path(d) / "one.txt"
            t2 = Path(d) / "two.PDF"
            t1.write_text("a", encoding="utf-8")
            t2.write_bytes(b"%PDF-1.4\n")
            state = ingest.collect_docs_state(d)
        self.assertEqual(state["file_count"], 2)
        self.assertIn("files", state)
        keys = list(state["files"].keys())
        self.assertEqual(len(keys), 2)
        for meta in state["files"].values():
            self.assertIn("size", meta)
            self.assertIn("mtime_ns", meta)

    def test_save_load_docs_state_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            state_path = str(Path(d) / "state.json")
            state = {"docs_dir": d, "file_count": 0, "files": {}}
            ingest.save_docs_state(state_path, state)
            loaded = ingest.load_saved_docs_state(state_path)
        self.assertEqual(loaded, state)

    def test_load_saved_docs_state_missing(self) -> None:
        self.assertIsNone(ingest.load_saved_docs_state("/nonexistent/state.json"))

    def test_load_saved_docs_state_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write("not json{{{")
            path = f.name
        try:
            self.assertIsNone(ingest.load_saved_docs_state(path))
        finally:
            os.unlink(path)

    def test_docs_changed_when_saved_differs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_text("v1", encoding="utf-8")
            state_path = str(Path(d) / "state.json")
            ingest.save_docs_state(state_path, ingest.collect_docs_state(d))
            Path(d, "b.txt").write_text("new", encoding="utf-8")
            self.assertTrue(ingest.docs_changed(d, state_path))

    def test_docs_changed_false_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.txt").write_text("same", encoding="utf-8")
            state_path = str(Path(d) / "state.json")
            ingest.save_docs_state(state_path, ingest.collect_docs_state(d))
            self.assertFalse(ingest.docs_changed(d, state_path))


class TestRuntime(unittest.TestCase):
    def test_setup_logging_sets_level(self) -> None:
        with patch("logging.basicConfig") as bc:
            runtime.setup_logging("debug")
        args, kwargs = bc.call_args
        self.assertEqual(kwargs["level"], logging.DEBUG)

    def test_setup_logging_invalid_falls_back_to_info(self) -> None:
        with patch("logging.basicConfig") as bc:
            runtime.setup_logging("not_a_real_level_name")
        _args, kwargs = bc.call_args
        self.assertEqual(kwargs["level"], logging.INFO)

    @patch("local_agent.runtime.model_supports_tools")
    @patch("local_agent.runtime.load_mcp_tools")
    @patch("local_agent.runtime.create_chat_agent")
    @patch("local_agent.runtime.get_or_create_vectorstore")
    @patch("local_agent.runtime.SentenceTransformerEmbeddings")
    @patch("local_agent.runtime.setup_logging")
    @patch("local_agent.runtime.load_settings")
    def test_load_chat_agent_wiring(
        self,
        mock_load_settings: MagicMock,
        mock_setup_logging: MagicMock,
        mock_st: MagicMock,
        mock_vs: MagicMock,
        mock_create: MagicMock,
        mock_load_mcp: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        settings = _minimal_settings(st_embed_model="m", st_encode_batch=16, st_device="cpu")
        mock_load_settings.return_value = settings
        emb = object()
        mock_st.return_value = emb
        vs = object()
        mock_vs.return_value = vs
        agent = object()
        mock_create.return_value = agent
        mock_load_mcp.return_value = []
        mock_supports.return_value = False

        out_agent, out_settings = runtime.load_chat_agent()

        self.assertIs(out_agent, agent)
        self.assertIs(out_settings, settings)
        mock_load_settings.assert_called_once_with()
        mock_setup_logging.assert_called_once_with(settings.log_level)
        mock_st.assert_called_once_with(
            model_name=settings.st_embed_model,
            encode_batch_size=settings.st_encode_batch,
            device=settings.st_device,
        )
        mock_vs.assert_called_once_with(settings, emb)
        mock_create.assert_called_once_with(
            vs, settings, tools=[], tools_supported=False
        )
        # Empty tools list short-circuits the capability probe.
        mock_supports.assert_not_called()

    @patch("local_agent.runtime.model_supports_tools")
    @patch("local_agent.runtime.load_mcp_tools")
    @patch("local_agent.runtime.create_chat_agent")
    @patch("local_agent.runtime.get_or_create_vectorstore")
    @patch("local_agent.runtime.SentenceTransformerEmbeddings")
    @patch("local_agent.runtime.setup_logging")
    @patch("local_agent.runtime.load_settings")
    def test_load_chat_agent_with_mcp_tools_and_supported_model(
        self,
        mock_load_settings: MagicMock,
        _setup_logging: MagicMock,
        _st: MagicMock,
        _vs_factory: MagicMock,
        mock_create: MagicMock,
        mock_load_mcp: MagicMock,
        mock_supports: MagicMock,
    ) -> None:
        settings = _minimal_settings(mcp_enabled=True, mcp_servers_json='{"x":{}}')
        mock_load_settings.return_value = settings
        tool_a = MagicMock()
        tool_a.name = "search_docs"
        mock_load_mcp.return_value = [tool_a]
        mock_supports.return_value = True

        runtime.load_chat_agent()

        mock_load_mcp.assert_called_once_with(settings)
        mock_supports.assert_called_once_with(
            settings.ollama_base_url, settings.ollama_chat_model
        )
        _, kwargs = mock_create.call_args
        self.assertEqual(kwargs["tools"], [tool_a])
        self.assertTrue(kwargs["tools_supported"])

    def test_load_mcp_tools_disabled_returns_empty(self) -> None:
        settings = _minimal_settings(mcp_enabled=False, mcp_servers_json='{"x":{}}')
        self.assertEqual(runtime.load_mcp_tools(settings), [])

    def test_load_mcp_tools_invalid_json_returns_empty(self) -> None:
        settings = _minimal_settings(mcp_enabled=True, mcp_servers_json="not-json")
        self.assertEqual(runtime.load_mcp_tools(settings), [])

    def test_load_mcp_tools_empty_object_returns_empty(self) -> None:
        settings = _minimal_settings(mcp_enabled=True, mcp_servers_json="{}")
        self.assertEqual(runtime.load_mcp_tools(settings), [])

    def test_load_mcp_tools_happy_path(self) -> None:
        settings = _minimal_settings(
            mcp_enabled=True,
            mcp_servers_json='{"docs": {"command": "x", "args": [], "transport": "stdio"}}',
        )
        fake_tool = MagicMock()
        fake_tool.name = "search_docs"
        fake_client = MagicMock()

        async def _get_tools():
            return [fake_tool]

        fake_client.get_tools.side_effect = _get_tools
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=fake_client,
        ) as ctor:
            tools = runtime.load_mcp_tools(settings)
        self.assertEqual(tools, [fake_tool])
        ctor.assert_called_once_with(
            {"docs": {"command": "x", "args": [], "transport": "stdio"}}
        )

    def test_model_supports_tools_returns_true_when_capability_present(self) -> None:
        class FakeResp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self) -> bytes:
                return self._payload

        payload = b'{"capabilities": ["completion", "tools"]}'
        with patch("urllib.request.urlopen", return_value=FakeResp(payload)):
            self.assertTrue(
                runtime.model_supports_tools("http://localhost:11434", "granite3.3")
            )

    def test_model_supports_tools_returns_false_on_network_error(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("boom"),
        ):
            self.assertFalse(
                runtime.model_supports_tools("http://localhost:11434", "granite3.3")
            )


class TestCli(unittest.TestCase):
    def test_main_eof_exits_without_run(self) -> None:
        fake_agent = MagicMock()
        settings = _minimal_settings()
        with (
            patch("local_agent.cli.load_chat_agent", return_value=(fake_agent, settings)),
            patch("builtins.input", side_effect=EOFError),
            patch("builtins.print") as print_mock,
        ):
            cli.main()
        fake_agent.run.assert_not_called()
        print_mock.assert_called_once_with("\nExiting.")

    def test_main_one_query_calls_run(self) -> None:
        fake_agent = MagicMock()
        fake_agent.run.return_value = "answer"
        settings = _minimal_settings()
        with (
            patch("local_agent.cli.load_chat_agent", return_value=(fake_agent, settings)),
            patch("builtins.input", side_effect=["question", EOFError]),
            patch("builtins.print") as print_mock,
        ):
            cli.main()
        fake_agent.run.assert_called_once_with("question")
        printed = [c.args[0] for c in print_mock.call_args_list]
        self.assertTrue(any("Answer:" in str(p) for p in printed))


class TestVectorstore(unittest.TestCase):
    def test_normalize_text_empty(self) -> None:
        self.assertEqual(vectorstore.SentenceTransformerEmbeddings.normalize_text(""), "")

    def test_normalize_text_strips_and_newlines(self) -> None:
        raw = "  a  \tb  \r\n  c\r  d  \n\n\n\nx  "
        out = vectorstore.SentenceTransformerEmbeddings.normalize_text(raw)
        self.assertNotIn("\r", out)
        self.assertNotIn("\x00", vectorstore.SentenceTransformerEmbeddings.normalize_text("a\x00b"))
        self.assertIn("a b", out)
        self.assertIn("c", out)
        self.assertIn("d", out)
        self.assertIn("x", out)

    def test_split_and_sanitize_truncates(self) -> None:
        docs = [Document(page_content="word " * 500, metadata={"source": "s"})]
        chunks = vectorstore.split_and_sanitize_documents(docs, max_chars=50)
        self.assertTrue(all(len(c.page_content) <= 50 for c in chunks))
        self.assertGreater(len(chunks), 0)

    def test_split_and_sanitize_drops_empty(self) -> None:
        docs = [Document(page_content="   \n\n  ", metadata={})]
        with self.assertRaises(ValueError) as ctx:
            vectorstore.split_and_sanitize_documents(docs, max_chars=100)
        self.assertIn("No valid chunks", str(ctx.exception))

    def test_build_faiss_ivfpq_batched_rejects_empty_chunks(self) -> None:
        emb = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            vectorstore.build_faiss_ivfpq_batched(
                [],
                emb,
                batch_size=8,
                num_threads=1,
                nlist=4,
                pq_m=4,
                pq_nbits=8,
                nprobe=1,
            )
        self.assertIn("empty", str(ctx.exception).lower())

    def test_build_faiss_ivfpq_batched_rejects_bad_pq_m(self) -> None:
        class StubEmb:
            encode_batch_size = 32

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # dim 127: not divisible by pq_m=64
                return [[0.0] * 127 for _ in texts]

        chunks = [Document(page_content="x", metadata={})]
        with self.assertRaises(ValueError) as ctx:
            vectorstore.build_faiss_ivfpq_batched(
                chunks,
                StubEmb(),  # type: ignore[arg-type]
                batch_size=8,
                num_threads=1,
                nlist=4,
                pq_m=64,
                pq_nbits=8,
                nprobe=1,
            )
        self.assertIn("pq_m", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
