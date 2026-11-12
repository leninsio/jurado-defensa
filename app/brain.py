"""OpenCode-backed brains for the jury simulator.

This module owns an app-isolated ``opencode serve`` child process and a thin
client for the OpenCode v2 HTTP API:

- ``POST /api/session`` creates a session bound to an agent, a model and an
  app-owned working directory.
- ``POST /api/session/{id}/prompt`` admits a user message.
- ``POST /api/experimental/session/{id}/wait`` blocks until the session is idle.
- ``GET /api/session/{id}/message`` returns messages newest first; the reply is
  the newest ``assistant`` message, joining its ``text`` content parts.
- ``POST /api/session/{id}/generate`` generates text from the session context.

Isolation matters: the user's global OpenCode config leaks instructions into
every agent, so the server runs with ``XDG_CONFIG_HOME`` pointing at an
app-owned directory that contains only the generated ``opencode.json``. Provider
credentials live under ``XDG_DATA_HOME`` and keep working.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash"
DEFAULT_CONTEXT_KEYS = ("tesis", "normas")

# Canonical order of the context bundle. It is emitted first, byte-identical
# for every role, so all OpenCode sessions share one provider cache prefix.
SHARED_CONTEXT_ORDER = ("tesis", "normas", "presentacion", "guia_estudio")

CONTEXT_TITLES = {
    "tesis": "Tesis (texto completo)",
    "normas": "Normas CUAM (TEG-600)",
    "presentacion": "Guion de la presentación (por lámina)",
    "guia_estudio": "Guía de estudio de la defensa",
}


class BrainError(RuntimeError):
    """Raised when the OpenCode brain cannot be prepared, started or queried."""


def free_port(host: str = "127.0.0.1") -> int:
    """Return a currently free TCP port on ``host``."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def split_model(reference: str) -> tuple[str, str]:
    """Split ``provider/model`` into its two halves."""

    provider, _, model = reference.partition("/")
    if not provider or not model:
        raise BrainError(f"invalid model reference: {reference!r} (expected provider/model)")
    return provider, model


def load_config(path: Path | str = REPO_ROOT / "config.json") -> dict:
    """Load the app configuration from ``path``."""

    config_path = Path(path).expanduser()
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BrainError(f"config not found: {config_path}") from exc


class Brain:
    """Lifecycle and client for the app-owned OpenCode server."""

    def __init__(
        self,
        config: dict,
        repo_root: Path | str | None = None,
        runtime_dir: Path | str | None = None,
        base_url: str | None = None,
        password: str | None = None,
    ) -> None:
        self.config = config
        self.repo_root = Path(repo_root).resolve() if repo_root else REPO_ROOT
        runtime = Path(runtime_dir) if runtime_dir else Path(config.get("runtime_dir", "runtime"))
        self.runtime_dir = runtime if runtime.is_absolute() else self.repo_root / runtime
        self.base_url = base_url.rstrip("/") if base_url else None
        self.password = password
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._owns_server = False
        self._sessions: dict[str, str] = {}

    # ------------------------------------------------------------------ paths

    @property
    def roles(self) -> dict[str, dict]:
        return self.config.get("roles") or {}

    @property
    def workdir(self) -> Path:
        """Empty working directory for sessions.

        It must live outside the user's home: OpenCode discovers ``AGENTS.md``
        by walking up from the session directory, and ``~/AGENTS.md`` leaks the
        user's global Engram/skills instructions into the judges.
        """

        configured = self.config.get("workdir")
        if configured:
            return Path(configured).expanduser()
        return Path(tempfile.gettempdir()) / "jurado-defensa" / "work"

    def runtime_paths(self) -> dict[str, Path]:
        return {
            "root": self.runtime_dir,
            "xdg": self.runtime_dir / "xdg",
            "prompts": self.runtime_dir / "prompts",
            "work": self.workdir,
            "log": self.runtime_dir / "serve.log",
        }

    def context_path(self, key: str) -> Path:
        context = self.config.get("context") or {}
        files = context.get("files") or {}
        if key not in files:
            raise BrainError(f"unknown context key: {key}")
        return self.repo_root / context.get("dir", "data/context") / files[key]

    def context_paths(self) -> dict[str, Path]:
        return {key: self.context_path(key) for key in self._context_files()}

    def _context_files(self) -> dict[str, str]:
        return (self.config.get("context") or {}).get("files") or {}

    def _role(self, role: str) -> dict:
        try:
            return self.roles[role]
        except KeyError as exc:
            available = ", ".join(sorted(self.roles))
            raise BrainError(f"unknown role {role!r}; available roles: {available}") from exc

    # ---------------------------------------------------------------- prompts

    def build_prompt(self, role: str) -> str:
        """Compose the full agent prompt for ``role`` from config.json.

        The large shared context block comes first and is byte-identical for
        every role: all OpenCode sessions then share one provider cache prefix.
        The role persona, tone and rules come after it.
        """

        role_config = self._role(role)
        name = role_config.get("name", role)
        title = role_config.get("role", role)
        configured = role_config.get("context") or list(DEFAULT_CONTEXT_KEYS)
        context_keys = [key for key in SHARED_CONTEXT_ORDER if key in configured]
        context_keys += [
            key for key in configured if key not in SHARED_CONTEXT_ORDER
        ]

        lines = [
            "## Contexto compartido",
            "",
            "Los archivos siguientes son tu única fuente de verdad. Léelos antes "
            "de responder y no contradigas lo que dicen.",
            "",
        ]
        for key in context_keys:
            lines += [
                f"### {CONTEXT_TITLES.get(key, key)}",
                "{file:" + str(self.context_path(key)) + "}",
                "",
            ]

        lines += [f"# {name} — {title}", ""]

        intro = role_config.get("intro")
        if intro:
            lines += [intro.format(name=name, role=title), ""]

        if role_config.get("persona"):
            lines += ["## Personalidad", role_config["persona"], ""]
        if role_config.get("tone"):
            lines += ["## Tono", role_config["tone"], ""]
        if role_config.get("questioning_style"):
            lines += ["## Estilo de preguntas", role_config["questioning_style"], ""]
        topics = role_config.get("favorite_topics") or []
        if topics:
            lines.append("## Temas favoritos")
            lines += [f"- {topic}" for topic in topics]
            lines.append("")

        kind = role_config.get("kind", "juez")
        rules = (self.config.get("rules") or {}).get(kind)
        if rules:
            lines += [rules.strip(), ""]

        return "\n".join(lines).rstrip() + "\n"

    def build_opencode_config(self) -> dict:
        """Build the app-owned ``opencode.json`` with one agent per role."""

        agents: dict[str, Any] = {}
        for role in self.roles:
            role_config = self.roles[role]
            agents[role] = {
                "description": f"{role_config.get('role', role)} — {role_config.get('name', role)}",
                "mode": "primary",
                "model": role_config.get("model") or DEFAULT_MODEL,
                "prompt": self.build_prompt(role),
                "permission": {"*": "deny"},
            }
        return {"$schema": "https://opencode.ai/config.json", "agent": agents}

    # ---------------------------------------------------------------- runtime

    def prepare_runtime(self, require_context: bool = True) -> dict[str, Path]:
        """Write the isolated OpenCode config and the composed prompts."""

        paths = self.runtime_paths()
        for key in ("xdg", "prompts", "work"):
            paths[key].mkdir(parents=True, exist_ok=True)

        if require_context:
            missing = [
                str(path) for path in self.context_paths().values() if not path.is_file()
            ]
            if missing:
                raise BrainError(
                    "missing context files: "
                    + ", ".join(missing)
                    + ". Run .venv/bin/python scripts/build_context.py first."
                )

        for role in self.roles:
            (paths["prompts"] / f"{role}.md").write_text(
                self.build_prompt(role), encoding="utf-8"
            )

        config_path = paths["xdg"] / "opencode" / "opencode.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.build_opencode_config(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths["config"] = config_path
        return paths

    def server_env(self, password: str) -> dict[str, str]:
        """Environment for the child server: isolated config, app password."""

        env = os.environ.copy()
        # The parent shell (or a delegating agent) may point OpenCode at another
        # config; drop those overrides so only the app-owned config is loaded.
        for key in (
            "OPENCODE_CONFIG",
            "OPENCODE_CONFIG_CONTENT",
            "OPENCODE_CONFIG_DIR",
            "OPENCODE_PERMISSION",
        ):
            env.pop(key, None)
        env["XDG_CONFIG_HOME"] = str(self.runtime_paths()["xdg"])
        env["OPENCODE_SERVER_PASSWORD"] = password
        return env

    def _binary(self) -> str:
        configured = (self.config.get("opencode") or {}).get("binary", "opencode")
        binary = shutil.which(configured) if not Path(configured).is_absolute() else configured
        if not binary:
            raise BrainError(f"opencode binary not found: {configured}")
        return binary

    # --------------------------------------------------------------- lifecycle

    def start(self, timeout: float | None = None) -> "Brain":
        """Start the app-owned ``opencode serve`` and wait until it answers."""

        if self.base_url:
            return self

        paths = self.prepare_runtime()
        settings = self.config.get("opencode") or {}
        host = settings.get("hostname", "127.0.0.1")
        port = free_port(host)
        self.password = secrets.token_urlsafe(24)
        env = self.server_env(self.password)

        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        with paths["log"].open("wb") as log:
            self._process = subprocess.Popen(
                [self._binary(), "serve", "--hostname", host, "--port", str(port)],
                cwd=str(self.repo_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._owns_server = True
        self.base_url = f"http://{host}:{port}"
        try:
            self._wait_until_ready(
                timeout if timeout is not None else settings.get("start_timeout_seconds", 60)
            )
        except BrainError:
            self.stop()
            raise
        return self

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error = "no response yet"
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise BrainError(
                    f"opencode serve exited with code {self._process.returncode}: "
                    + self._log_tail()
                )
            try:
                response = self._http().get("/api/agent", timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"{response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise BrainError(f"opencode serve did not become ready in {timeout}s ({last_error})")

    def _log_tail(self, lines: int = 20) -> str:
        log_path = self.runtime_paths()["log"]
        if not log_path.is_file():
            return "(no log)"
        return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def stop(self) -> None:
        """Stop the managed server, if any. Safe to call more than once."""

        if self._client is not None:
            self._client.close()
            self._client = None
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            self._signal(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
                process.wait(timeout=5)
        if self._owns_server:
            self.base_url = None
            self._owns_server = False

    @staticmethod
    def _signal(process: subprocess.Popen, sig: int) -> None:
        """Signal the whole process group so no opencode child survives."""

        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def __enter__(self) -> "Brain":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------ client

    def _http(self) -> httpx.Client:
        if self._client is None:
            if not self.base_url or not self.password:
                raise BrainError("brain is not started: call start() first")
            settings = self.config.get("opencode") or {}
            self._client = httpx.Client(
                base_url=self.base_url,
                auth=(settings.get("user", "opencode"), self.password),
                timeout=httpx.Timeout(
                    settings.get("request_timeout_seconds", 30),
                    read=settings.get("wait_timeout_seconds", 300),
                ),
            )
        return self._client

    def _request(self, method: str, path: str, expected: tuple[int, ...] = (200,), **kwargs):
        try:
            response = self._http().request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BrainError(f"{method} {path} failed: {exc}") from exc
        if response.status_code not in expected:
            raise BrainError(
                f"{method} {path} returned {response.status_code}: {response.text[:500]}"
            )
        return response

    def create_session(self, role: str, title: str | None = None) -> str:
        """Create a session bound to ``role`` and return its id."""

        role_config = self._role(role)
        provider, model = split_model(role_config.get("model") or DEFAULT_MODEL)
        model_ref: dict[str, str] = {"providerID": provider, "id": model}
        variant = role_config.get("variant")
        if variant:
            model_ref["variant"] = str(variant)
        self.workdir.mkdir(parents=True, exist_ok=True)
        body = {
            "title": title or f"{role} · {datetime.now():%Y-%m-%d %H:%M}",
            "agent": role,
            "model": model_ref,
            "location": {"directory": str(self.workdir)},
        }
        payload = self._request("POST", "/api/session", json=body).json()
        session_id = (payload.get("data") or {}).get("id")
        if not session_id:
            raise BrainError(f"session creation returned no id: {payload}")
        return str(session_id)

    def ask(
        self, session_id: str, text: str, files: list[dict] | None = None
    ) -> str:
        """Send ``text`` to the session and return the assistant reply text.

        ``files`` are OpenCode prompt attachments (``{"uri", "name"}``); a
        ``data:image/...;base64`` URI lets a vision model see the image.
        """

        body: dict[str, Any] = {"text": text}
        if files:
            body["files"] = files
        prompt = self._request(
            "POST", f"/api/session/{session_id}/prompt", json=body
        ).json()
        user_created = int(((prompt.get("data") or {}).get("time") or {}).get("created") or 0)

        self._request(
            "POST",
            f"/api/experimental/session/{session_id}/wait",
            json={},
            expected=(200, 204),
        )

        payload = self._request(
            "GET",
            f"/api/session/{session_id}/message",
            params={"type": "assistant", "limit": "1", "order": "desc"},
        ).json()
        messages = payload.get("data") or []
        if not messages:
            raise BrainError(f"session {session_id} has no assistant reply")
        message = messages[0]
        created = int((message.get("time") or {}).get("created") or 0)
        if user_created and created < user_created:
            raise BrainError(
                f"stale assistant reply in session {session_id}: "
                f"reply created {created} before prompt {user_created}"
            )

        parts = [
            str(part.get("text", "")).strip()
            for part in message.get("content") or []
            if part.get("type") == "text"
        ]
        reply = "\n".join(part for part in parts if part).strip()
        if not reply:
            raise BrainError(f"session {session_id} returned an empty reply")
        return reply

    def generate(self, role: str, prompt: str, title: str | None = None) -> str:
        """Generate text for ``role`` without adding a conversation turn."""

        self._role(role)
        session_id = self._sessions.get(role)
        if not session_id:
            session_id = self.create_session(role, title)
            self._sessions[role] = session_id
        payload = self._request(
            "POST", f"/api/session/{session_id}/generate", json={"prompt": prompt}
        ).json()
        text = str((payload.get("data") or {}).get("text") or "").strip()
        if not text:
            raise BrainError(f"generate for role {role!r} returned no text")
        return text
