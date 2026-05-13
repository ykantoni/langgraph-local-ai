"""Tests for the FastMCP-based local document search server."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from local_agent import mcp_server


class TestBuildMcpServer(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets a fresh module-level cache.
        mcp_server._vectorstore = None
        mcp_server._settings = None

    def tearDown(self) -> None:
        mcp_server._vectorstore = None
        mcp_server._settings = None

    def _get_search_tool(self):
        server = mcp_server.build_mcp_server(name="local-docs-test")
        tools = asyncio.run(server.list_tools())
        names = [t.name for t in tools]
        self.assertIn("search_docs", names)
        return server, next(t for t in tools if t.name == "search_docs")

    def test_search_docs_tool_is_registered(self) -> None:
        server, tool = self._get_search_tool()
        self.assertEqual(tool.name, "search_docs")
        # The docstring becomes the tool description.
        self.assertIn("local FAISS", (tool.description or ""))
        self.assertIsNotNone(server)

    def test_search_docs_invokes_vectorstore_lazily(self) -> None:
        fake_vs = MagicMock()
        fake_vs.similarity_search.return_value = [
            Document(page_content="alpha", metadata={}),
            Document(page_content="beta", metadata={}),
        ]

        with patch.object(mcp_server, "_get_vectorstore", return_value=fake_vs):
            server = mcp_server.build_mcp_server(name="local-docs-test")
            result = asyncio.run(
                server.call_tool("search_docs", {"query": "hello", "k": 2})
            )

        fake_vs.similarity_search.assert_called_once_with("hello", k=2)
        # FastMCP returns (content_blocks, structured) for call_tool.
        if isinstance(result, tuple):
            content_blocks = result[0]
        else:
            content_blocks = result
        text_parts = [getattr(b, "text", "") for b in content_blocks]
        joined = "\n".join(text_parts)
        self.assertIn("alpha", joined)
        self.assertIn("beta", joined)


if __name__ == "__main__":
    unittest.main()
