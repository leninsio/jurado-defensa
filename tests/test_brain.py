"""Tests for app/brain.py against a fake OpenCode v2 server."""

from __future__ import annotations

import base64
import copy
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from app import brain as brain_module
from app.brain import Brain, BrainError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))

ROLES = ("juez_metodologo", "juez_tecnico", "juez_presidente", "evaluador", "tutor")


@dataclass
class RecordedRequest:
    method: str
    path: str
    query: dict[str, list[str]]
    body: bytes
    headers: dict[str, str]

    def json(self):
        return json.loads(self.body or b"{}")


@dataclass
class FakeOpenCode:
    """Minimal reproduction of the OpenCode v2 endpoints used by the brain."""

    password: str = "secret"
    user: str = "opencode"
    session_id: str = "ses_fake"
    user_created: int = 1000
    assistant_created: int = 1001
    reply_parts: list[dict] = field(
        default_factory=lambda: [
            {"type": "reasoning", "text": "pensando"},
            {"type": "text", "text": "La tesis propone un sistema."},
        ]
    )
    generate_text: str = '{"slide": 1, "fortalezas": [], "correcciones": [], "frase_sugerida": ""}'
    requests: list[RecordedRequest] = field(default_factory=list)
    fail_path: str | None = None
    fail_status: int = 500

    def start(self) -> None:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # keep pytest output clean
                pass

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization", "")
                expected = "Basic " + base64.b64encode(
                    f"{fake.user}:{fake.password}".encode()
                ).decode()
                return header == expected

            def _reply(self, status: int, payload: dict | None = None) -> None:
                body = b"" if payload is None else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _record(self, method: str) -> RecordedRequest:
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                request = RecordedRequest(
                    method=method,
                    path=parsed.path,
                    query=parse_qs(parsed.query),
                    body=body,
                    headers=dict(self.headers),
                )
                fake.requests.append(request)
                return request

            def _dispatch(self, method: str) -> None:
                request = self._record(method)
                if not self._authorized():
                    self._reply(401, {"error": "unauthorized"})
                    return
                if fake.fail_path and request.path.startswith(fake.fail_path):
                    self._reply(fake.fail_status, {"error": "boom"})
                    return

                if method == "POST" and request.path == "/api/session":
                    self._reply(200, {"data": {"id": fake.session_id}})
                    return
                if method == "POST" and re.fullmatch(
                    r"/api/session/[^/]+/prompt", request.path
                ):
                    self._reply(
                        200,
                        {
                            "data": {
                                "id": "msg_user",
                                "type": "user",
                                "time": {"created": fake.user_created},
                            }
                        },
                    )
                    return
                if method == "POST" and re.fullmatch(
                    r"/api/experimental/session/[^/]+/wait", request.path
                ):
                    self._reply(204)
                    return
                if method == "GET" and re.fullmatch(
                    r"/api/session/[^/]+/message", request.path
                ):
                    self._reply(
                        200,
                        {
                            "data": [
                                {
                                    "id": "msg_assistant",
                                    "type": "assistant",
                                    "time": {"created": fake.assistant_created},
                                    "content": fake.reply_parts,
                                }
                            ]
                        },
                    )
                    return
                if method == "POST" and re.fullmatch(
                    r"/api/session/[^/]+/generate", request.path
                ):
                    self._reply(200, {"data": {"text": fake.generate_text}})
                    return
                self._reply(404, {"error": "not found", "path": request.path})

            def do_GET(self):
                self._dispatch("GET")

            def do_POST(self):
                self._dispatch("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def last(self, method: str, path: str) -> RecordedRequest:
        matches = [r for r in self.requests if r.method == method and r.path == path]
        assert matches, f"no {method} {path} in {[ (r.method, r.path) for r in self.requests ]}"
        return matches[-1]


@pytest.fixture
def fake_server():
    server = FakeOpenCode()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def config() -> dict:
    return copy.deepcopy(CONFIG)


def seed_context(root: Path) -> None:
    context = root / "data" / "context"
    context.mkdir(parents=True, exist_ok=True)
    for name in ("tesis.txt", "normas_cuam.txt", "presentacion.txt", "guia_estudio.txt"):
        (context / name).write_text(f"contenido de {name}\n", encoding="utf-8")


@pytest.fixture
def local_brain(tmp_path, config):
    seed_context(tmp_path)
    return Brain(config, repo_root=tmp_path, runtime_dir=tmp_path / "runtime")


@pytest.fixture
def client_brain(tmp_path, config, fake_server):
    seed_context(tmp_path)
    instance = Brain(
        config,
        repo_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        base_url=fake_server.url,
        password=fake_server.password,
    )
    yield instance
    instance.stop()


def test_create_session_uses_role_agent_model_and_workdir(client_brain, fake_server):
    session_id = client_brain.create_session("juez_presidente")

    assert session_id == "ses_fake"
    body = fake_server.last("POST", "/api/session").json()
    assert body["agent"] == "juez_presidente"
    assert body["model"] == {
        "providerID": "opencode-go",
        "id": "deepseek-v4.1-flash",
        "variant": "low",
    }
    assert body["location"]["directory"] == str(client_brain.workdir)
    assert client_brain.workdir.is_dir()


def test_create_session_omits_the_variant_for_non_judge_roles(client_brain, fake_server):
    client_brain.create_session("evaluador")

    body = fake_server.last("POST", "/api/session").json()
    assert "variant" not in body["model"]


def test_ask_returns_only_text_parts_and_waits(client_brain, fake_server):
    fake_server.reply_parts = [
        {"type": "reasoning", "text": "pensando"},
        {"type": "text", "text": "La tesis propone"},
        {"type": "text", "text": "un sistema de inventario."},
    ]

    reply = client_brain.ask("ses_fake", "¿Cuál es el objetivo general?")

    assert reply == "La tesis propone\nun sistema de inventario."
    assert fake_server.last("POST", "/api/session/ses_fake/prompt").json() == {
        "text": "¿Cuál es el objetivo general?"
    }
    fake_server.last("POST", "/api/experimental/session/ses_fake/wait")
    query = fake_server.last("GET", "/api/session/ses_fake/message").query
    assert query["type"] == ["assistant"]
    assert query["limit"] == ["1"]
    assert query["order"] == ["desc"]


def test_ask_sends_file_attachments(client_brain, fake_server):
    files = [{"uri": "data:image/png;base64,AAAA", "name": "lamina-1.png"}]

    client_brain.ask("ses_fake", "¿Qué ves?", files=files)

    assert fake_server.last("POST", "/api/session/ses_fake/prompt").json() == {
        "text": "¿Qué ves?",
        "files": files,
    }


def test_ask_rejects_a_stale_assistant_reply(client_brain, fake_server):
    fake_server.user_created = 5000
    fake_server.assistant_created = 4000

    with pytest.raises(BrainError, match="stale"):
        client_brain.ask("ses_fake", "hola")


def test_generate_returns_the_text_field(client_brain, fake_server):
    fake_server.generate_text = '{"slide": 3, "fortalezas": ["clara"]}'

    assert client_brain.generate("evaluador", "evalúa la lámina 3") == fake_server.generate_text
    assert fake_server.last("POST", "/api/session/ses_fake/generate").json() == {
        "prompt": "evalúa la lámina 3"
    }


def test_unknown_role_lists_available_roles(client_brain):
    with pytest.raises(BrainError, match="juez_metodologo"):
        client_brain.create_session("juez_fantasma")


def test_http_errors_become_brain_errors(client_brain, fake_server):
    fake_server.fail_path = "/api/session"

    with pytest.raises(BrainError, match="500"):
        client_brain.create_session("tutor")


def test_wrong_password_is_reported_as_unauthorized(tmp_path, config, fake_server):
    seed_context(tmp_path)
    instance = Brain(
        config,
        repo_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        base_url=fake_server.url,
        password="wrong-password",
    )

    with pytest.raises(BrainError, match="401"):
        instance.create_session("tutor")


def test_stop_is_idempotent_without_a_managed_process(client_brain):
    client_brain.stop()
    client_brain.stop()


def test_build_opencode_config_isolates_and_distinguishes_agents(local_brain):
    generated = local_brain.build_opencode_config()
    agents = generated["agent"]

    assert set(agents) == set(ROLES)
    prompts = {}
    for name, agent in agents.items():
        assert agent["mode"] == "primary"
        assert agent["permission"] == {"*": "deny"}
        assert agent["model"] == "opencode-go/deepseek-v4.1-flash"
        prompts[name] = agent["prompt"]
    assert len(set(prompts.values())) == len(ROLES)

    president = prompts["juez_presidente"]
    assert "Carmen Rojas" in president
    assert "{file:" + str(local_brain.context_path("tesis")) + "}" in president
    assert "{file:" + str(local_brain.context_path("normas")) + "}" in president
    assert "una sola pregunta" in president.lower()

    assert "fortalezas" in prompts["evaluador"]
    assert "frase_sugerida" in prompts["evaluador"]
    assert "JSON" in prompts["evaluador"]
    assert "70 palabras" in prompts["tutor"]


def test_build_prompt_puts_the_shared_context_before_the_role_persona(local_brain):
    prompts = {role: local_brain.build_prompt(role) for role in ROLES}

    def shared_prefix(prompt: str) -> str:
        index = prompt.find("\n# ")
        assert index != -1, "the role heading must come after the shared context"
        return prompt[: index + 1]

    prefixes = {role: shared_prefix(prompt) for role, prompt in prompts.items()}
    assert len(set(prefixes.values())) == 1, "every role must share one byte-identical prefix"

    shared = next(iter(prefixes.values()))
    markers = [
        "{file:" + str(local_brain.context_path(key)) + "}"
        for key in ("tesis", "normas", "presentacion", "guia_estudio")
    ]
    positions = [shared.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "Carmen Rojas" not in shared
    assert "fortalezas" not in shared


def test_every_role_configures_the_same_context_superset(local_brain):
    expected = ["tesis", "normas", "presentacion", "guia_estudio"]
    for role, settings in local_brain.roles.items():
        assert settings.get("context") == expected, role


def test_prepare_runtime_writes_config_and_prompts(local_brain):
    paths = local_brain.prepare_runtime()

    config_path = paths["xdg"] / "opencode" / "opencode.json"
    assert config_path.is_file()
    generated = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(generated["agent"]) == set(ROLES)

    prompt_files = sorted(p.name for p in paths["prompts"].glob("*.md"))
    assert prompt_files == sorted(f"{role}.md" for role in ROLES)
    assert "Carmen Rojas" in (paths["prompts"] / "juez_presidente.md").read_text(
        encoding="utf-8"
    )
    assert paths["work"].is_dir()


def test_server_env_isolates_xdg_and_carries_the_password(local_brain, monkeypatch):
    paths = local_brain.prepare_runtime()
    monkeypatch.setenv("OPENCODE_CONFIG", "/tmp/leaky-opencode.json")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "{}")

    env = local_brain.server_env("pw")

    assert env["XDG_CONFIG_HOME"] == str(paths["xdg"])
    assert env["OPENCODE_SERVER_PASSWORD"] == "pw"
    assert env["PATH"]
    assert "OPENCODE_CONFIG" not in env
    assert "OPENCODE_CONFIG_CONTENT" not in env


def test_missing_context_asks_for_build_context(tmp_path, config):
    instance = Brain(config, repo_root=tmp_path, runtime_dir=tmp_path / "runtime")

    with pytest.raises(BrainError, match="build_context"):
        instance.prepare_runtime()


def test_free_port_returns_a_bindable_port():
    port = brain_module.free_port("127.0.0.1")
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_workdir_lives_outside_home_so_global_agents_do_not_leak(tmp_path, config):
    instance = Brain(config, repo_root=tmp_path, runtime_dir=tmp_path / "runtime")

    assert Path.home() not in instance.workdir.parents
    assert instance.workdir != tmp_path / "runtime" / "work"
