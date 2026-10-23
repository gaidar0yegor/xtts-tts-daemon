#!/usr/bin/env python3
"""Contract test for xtts_daemon.py — runs the daemon in --mock mode.

Asserts the HTTP surface the Hermes WebUI ``xtts-tts`` extension depends on:
health, CORS preflight, a RIFF/WAVE POST at 24 kHz mono, long-text splitting,
and the 400/415 rejections. No model, no GPU, no network beyond loopback.

    python3 test_contract.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGIN = "http://127.0.0.1:8787"  # pretend WebUI origin: different port, same box
DAEMON = [sys.executable, str(HERE / "xtts_daemon.py"), "--mock"]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post(base: str, body: bytes, content_type: str = "application/json"):
    request = urllib.request.Request(
        base + "/",
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Origin": ORIGIN},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def wait_ready(base: str, process: subprocess.Popen, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit("daemon exited early; see its stderr above")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            time.sleep(0.15)
    raise SystemExit("daemon did not answer /health in %.0fs" % timeout)


results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print("ok   %s" % name)
    except Exception as error:  # noqa: BLE001 - the point is to report any failure
        results.append((name, False, str(error)))
        print("FAIL %s\n     %s" % (name, error))


def main() -> int:
    port = free_port()
    base = "http://127.0.0.1:%d" % port
    process = subprocess.Popen(DAEMON + ["--port", str(port)])
    try:
        health = wait_ready(base, process)

        def health_payload() -> None:
            assert health.get("ok") is True, health
            assert health.get("mock") is True, health
            assert "xtts_v2" in health.get("model", ""), health

        def preflight() -> None:
            request = urllib.request.Request(
                base + "/", method="OPTIONS",
                headers={
                    "Origin": ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 204, response.status
                headers = response.headers
                assert headers["Access-Control-Allow-Origin"] in (ORIGIN, "*"), headers["Access-Control-Allow-Origin"]
                assert "POST" in headers["Access-Control-Allow-Methods"]
                assert "OPTIONS" in headers["Access-Control-Allow-Methods"]
                assert "content-type" in headers["Access-Control-Allow-Headers"].lower()
                assert headers["Vary"] == "Origin", headers["Vary"]

        def synthesize_wav() -> None:
            payload = json.dumps(
                {"text": "Hello there, sir. How are you today?", "language": "en",
                 "speaker": "Viktor Menelaos"}
            ).encode("utf-8")
            status, headers, body = post(base, payload)
            assert status == 200, (status, body[:200])
            assert headers.get("Content-Type") == "audio/wav", headers
            assert body[:4] == b"RIFF" and body[8:12] == b"WAVE", body[:16]
            with wave.open(io.BytesIO(body)) as handle:
                assert handle.getnchannels() == 1, handle.getnchannels()
                assert handle.getframerate() == 24000, handle.getframerate()
                assert handle.getsampwidth() == 2, handle.getsampwidth()
                short_frames = handle.getnframes()
            # Same text, one long paragraph: splitting must yield a longer body.
            long_text = " ".join(["This is sentence number %d." % i for i in range(12)])
            status, _, long_body = post(
                base, json.dumps({"text": long_text, "language": "en"}).encode("utf-8")
            )
            assert status == 200, status
            with wave.open(io.BytesIO(long_body)) as handle:
                assert handle.getnframes() > short_frames * 3, (handle.getnframes(), short_frames)

        def rejects_empty_text() -> None:
            status, _, body = post(base, json.dumps({"text": "   "}).encode("utf-8"))
            assert status == 400, (status, body)
            assert b"empty text" in body, body

        def rejects_non_json() -> None:
            status, _, body = post(base, b"text=hello", content_type="text/plain")
            assert status == 415, (status, body)

        def rejects_bad_json() -> None:
            status, _, body = post(base, b"{not json")
            assert status == 400, (status, body)

        def health_speakers_shape() -> None:
            with urllib.request.urlopen(base + "/health", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert isinstance(payload.get("speakers"), list), payload

        check("GET /health reports the model and mock flag", health_payload)
        check("OPTIONS / answers the CORS preflight", preflight)
        check("POST / returns 24 kHz mono RIFF/WAVE and splits long text", synthesize_wav)
        check("POST / rejects empty text with 400", rejects_empty_text)
        check("POST / rejects a non-JSON content type with 415", rejects_non_json)
        check("POST / rejects malformed JSON with 400", rejects_bad_json)
        check("GET /health returns a speakers list", health_speakers_shape)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    failed = [name for name, ok, _ in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
