"""Tests for the mcp-local-rag launcher (supergateway + streamable HTTP)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from local_agent import mcp_server


class TestMcpLocalRagLauncher(unittest.TestCase):
    def test_build_local_rag_env_maps_docs_and_index_paths(self) -> None:
        settings = MagicMock()
        settings.docs_dir = "./docs"
        settings.faiss_index_dir = "./faiss_index"

        with patch.dict(os.environ, {}, clear=False):
            env = mcp_server.build_local_rag_env(settings)

        self.assertEqual(env["BASE_DIR"], os.path.abspath("./docs"))
        self.assertTrue(env["DB_PATH"].endswith(os.path.join("mcp_local_rag", "lancedb")))
        self.assertTrue(env["CACHE_DIR"].endswith(os.path.join("mcp_local_rag", "models")))

    def test_build_streamable_http_command_uses_supergateway(self) -> None:
        cmd = mcp_server.build_streamable_http_command(
            port=8765,
            path="/mcp",
            stdio_command="npx -y mcp-local-rag",
            stateful=True,
            session_timeout_ms=60000,
        )
        self.assertTrue(cmd[0].lower().endswith("npx") or cmd[0].lower().endswith("npx.cmd"))
        self.assertIn("supergateway", cmd)
        self.assertIn("--outputTransport", cmd)
        self.assertIn("streamableHttp", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("8765", cmd)
        self.assertIn("--streamableHttpPath", cmd)
        self.assertIn("/mcp", cmd)
        self.assertIn("--stateful", cmd)
        self.assertIn("--sessionTimeout", cmd)
        self.assertIn("60000", cmd)

    def test_main_runs_sync_then_streamable_http(self) -> None:
        settings = MagicMock()
        settings.docs_dir = "./docs"
        settings.faiss_index_dir = "./faiss_index"

        with (
            patch.object(mcp_server, "load_settings", return_value=settings),
            patch.object(mcp_server, "_require_npx"),
            patch.object(mcp_server, "run_sync", return_value=0) as sync_mock,
            patch.object(mcp_server, "run_streamable_http", return_value=0) as http_mock,
            patch.dict(
                os.environ,
                {"MCP_TRANSPORT": "streamable-http", "MCP_AUTO_SYNC": "1"},
                clear=False,
            ),
        ):
            mcp_server.main()

        sync_mock.assert_called_once()
        http_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
