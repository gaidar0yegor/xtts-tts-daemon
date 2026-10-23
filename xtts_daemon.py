#!/usr/bin/env python3
"""Minimal XTTS v2 HTTP daemon for the Hermes WebUI ``xtts-tts`` extension.

Contract (what the browser extension POSTs and expects back)::

    POST /                                  Content-Type: application/json
    {"text": "...", "language": "ru", "speaker": "Viktor Menelaos"}
    -> 200 audio/wav  (RIFF/WAVE, 24 kHz, 16-bit mono)

    OPTIONS /                               CORS preflight -> 204
    GET  /health                            -> 200 {"ok": true, ...}

Why the CORS headers matter: the WebUI page and this daemon are different
origins (different ports), so the browser preflights every POST. A daemon that
does not answer OPTIONS with ``Access-Control-Allow-*`` makes the WebUI show a
bare "Failed to fetch" and the extension's own message that names CORS.

The daemon listens on loopback only by default. The port, not the Origin
header, is the security boundary: anything that can reach 127.0.0.1:8120 can
synthesize, so do not bind it to a public interface.

Run with ``--mock`` to serve a short tone instead of a model: that path needs
no GPU, no model download, and is what the HTTP contract tests use.

Install (XTTS v2 needs Python 3.9-3.11; Coqui's fork lives on PyPI as
``coqui-tts``)::

    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install coqui-tts
    .venv/bin/python xtts_daemon.py --port 8120 --speaker "Viktor Menelaos"

The first non-mock request downloads/loads ``tts_models/multilingual/
multi-dataset/xtts_v2`` (~1.8 GB) and takes a while.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import struct
import sys
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMPLE_RATE = 24000
MAX_BODY_BYTES = 1 << 20          # 1 MiB of text is far more than a reply
MAX_TEXT_CHARS = 20000
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
MAX_CHUNK_CHARS = 220             # XTTS degrades on long single chunks

ALLOWED_METHODS = "POST, OPTIONS"
ALLOWED_HEADERS = "Content-Type"
LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

_state = {
    "tts": None,
    "device": "cpu",
    "speaker": None,
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "mock": False,
    "loading": False,
}
_model_lock = threading.Lock()
_synth_lock = threading.Lock()     # XTTS is not thread-safe


def log(message: str) -> None:
    sys.stderr.write("[xtts-daemon] %s\n" % message)
    sys.stderr.flush()


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries, then hard-wrap anything still too long."""
    chunks: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def load_model() -> None:
    """Import and load the model once, under the model lock."""
    if _state["tts"] is not None or _state["mock"]:
        return
    with _model_lock:
        if _state["tts"] is not None:
            return
        _state["loading"] = True
        try:
            log("loading %s on %s (first request; expect a download on a cold cache)"
                % (_state["model_name"], _state["device"]))
            from TTS.api import TTS  # imported here so --mock needs no Coqui install

            tts = TTS(_state["model_name"]).to(_state["device"])
            _state["tts"] = tts
            log("model ready")
        finally:
            _state["loading"] = False


def speakers() -> list[str]:
    tts = _state["tts"]
    try:
        model = getattr(tts, "synthesizer", None)
        model = getattr(model, "tts_model", None)
        names = getattr(model, "speaker_manager", None)
        names = getattr(names, "name_to_id", None)
        if isinstance(names, dict):
            return sorted(str(name) for name in names)
    except Exception:  # pragma: no cover - defensive, API drift between versions
        pass
    return []


def mock_chunk(text: str) -> bytes:
    """A short tone whose length tracks the text, so playback is auditable."""
    seconds = min(2.5, 0.25 + 0.02 * len(text))
    frames = int(SAMPLE_RATE * seconds)
    samples = []
    for i in range(frames):
        # Fade in/out so the WAV does not click.
        envelope = min(1.0, i / 400.0, (frames - i) / 400.0)
        samples.append(0.25 * envelope * math.sin(2 * math.pi * 220.0 * i / SAMPLE_RATE))
    return struct.pack("<%dh" % len(samples), *(int(s * 32767) for s in samples))


def synth_chunk(text: str, language: str, speaker: str) -> list[float]:
    tts = _state["tts"]
    with _synth_lock:
        return tts.tts(text=text, speaker=speaker, language=language)


def synthesize(text: str, language: str, speaker: str) -> bytes:
    language = language if language in LANGUAGES else "en"
    chunks = split_text(text) or [text]
    if _state["mock"]:
        pcm = b"".join(mock_chunk(chunk) for chunk in chunks)
        return pcm_to_wav(pcm)

    load_model()
    known = speakers()
    if known and speaker not in known:
        raise ValueError(
            "unknown speaker %r for this XTTS v2 model; try one of: %s"
            % (speaker, ", ".join(known[:8]) + (" …" if len(known) > 8 else ""))
        )

    samples: list[float] = []
    for chunk in chunks:
        samples.extend(synth_chunk(chunk, language, speaker))
    pcm = struct.pack("<%dh" % len(samples),
                      *(max(-32767, min(32767, int(s * 32767))) for s in samples))
    return pcm_to_wav(pcm)


def pcm_to_wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "xtts-daemon/0.1"
    protocol_version = "HTTP/1.1"

    # ── CORS ────────────────────────────────────────────────────────────
    def cors_headers(self) -> None:
        # Echo the caller's origin: the request comes from the WebUI page on a
        # different port. Loopback binding is the security boundary, so echoing
        # is safe here and keeps the daemon usable from any local WebUI port.
        origin = self.headers.get("Origin")
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", ALLOWED_METHODS)
        self.send_header("Access-Control-Allow-Headers", ALLOWED_HEADERS)
        self.send_header("Access-Control-Max-Age", "600")
        if origin:
            self.send_header("Vary", "Origin")

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_wav(self, wav: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.cors_headers()
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(wav)

    # ── routes ──────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server naming)
        self.send_response(HTTPStatus.NO_CONTENT)
        self.cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            self.send_json(HTTPStatus.OK, {
                "ok": True,
                "mock": bool(_state["mock"]),
                "model": _state["model_name"],
                "device": _state["device"],
                "loaded": _state["tts"] is not None,
                "loading": bool(_state["loading"]),
                "speakers": speakers()[:8],
            })
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/", "/tts", "/api/tts"):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                           {"error": "expected Content-Type: application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "bad Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                           {"error": "body must be 1 byte - %d bytes" % MAX_BODY_BYTES})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON: %s" % error})
            return
        if not isinstance(payload, dict):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON object"})
            return

        text = str(payload.get("text") or "").replace("\x00", "").strip()
        if not text:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "empty text"})
            return
        if len(text) > MAX_TEXT_CHARS:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                           {"error": "text too long (max %d chars)" % MAX_TEXT_CHARS})
            return

        language = str(payload.get("language") or "en").strip().lower()
        speaker = str(payload.get("speaker") or "").strip() or "Viktor Menelaos"

        started = time.monotonic()
        try:
            wav = synthesize(text, language, speaker)
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # model/backend failure: say so, never 200
            log("synthesis failed: %r" % (error,))
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": "synthesis failed: %s" % error})
            return

        log("%d chars, %s, %s -> %d bytes WAV in %.1fs"
            % (len(text), language, speaker, len(wav), time.monotonic() - started))
        self.send_wav(wav)

    def log_message(self, fmt: str, *args) -> None:  # keep one tidy line per request
        log("%s %s" % (self.address_string(), fmt % args))


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal XTTS v2 HTTP daemon.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; do not expose publicly)")
    parser.add_argument("--port", type=int, default=8120, help="TCP port (default 8120)")
    parser.add_argument("--speaker", default="Viktor Menelaos",
                        help="default speaker name (the extension sends its own)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--model", default=_state["model_name"],
                        help="Coqui model name (default xtts_v2 multilingual)")
    parser.add_argument("--mock", action="store_true",
                        help="serve a tone instead of a model (no GPU, no download)")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log("WARNING: binding %s exposes an unauthenticated TTS port; loopback only "
            "unless you really mean it" % args.host)

    _state.update(device=args.device, model_name=args.model, mock=bool(args.mock))
    _state["speaker"] = args.speaker

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log("listening on http://%s:%d/ (mock=%s, device=%s)" % (args.host, args.port, args.mock, args.device))
    if not args.mock:
        # Warm the model in the background: the socket has to answer /health (and
        # a request that arrives mid-load has to queue on the model lock) rather
        # than block until the weights are in memory.
        def warm() -> None:
            try:
                load_model()
            except Exception as error:  # noqa: BLE001 - surfaced on the request instead
                log("model load failed at startup (%r); will retry on the first request" % (error,))

        threading.Thread(target=warm, name="xtts-warmup", daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
