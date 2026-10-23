# xtts-tts-daemon

Minimal HTTP daemon that serves [Coqui XTTS v2](https://github.com/coqui-ai/TTS)
as a WAV endpoint for the Hermes WebUI extension
[`xtts-tts`](https://github.com/hermes-webui/hermes-webui-extensions/tree/main/extensions/xtts-tts).

It is deliberately small: stdlib `http.server` in front of `TTS.api`, one worker
lock around the model, sentence splitting for long text, and the CORS headers a
browser needs to POST to a different port. If you already run your own daemon
that speaks the [contract](#http-contract) below, you do not need this one.

## Install

XTTS v2 needs Python 3.9–3.11 (Coqui's fork is published as `coqui-tts`), a few
GB of disk for torch, and ~1.8 GB of model weights on first synthesis.

```bash
git clone https://github.com/gaidar0yegor/xtts-tts-daemon
cd xtts-tts-daemon
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install coqui-tts
```

GPU: install the CUDA build of torch for your driver first, then
`pip install coqui-tts`, and start the daemon with `--device cuda`.

## Run

```bash
.venv/bin/python xtts_daemon.py --port 8120 --speaker "Viktor Menelaos"
```

```
[xtts-daemon] listening on http://127.0.0.1:8120/ (mock=False, device=cpu)
[xtts-daemon] loading tts_models/multilingual/multi-dataset/xtts_v2 on cpu (...)
[xtts-daemon] model ready
```

The first start loads/downloads the model (slow). The daemon binds `127.0.0.1`
by default: **do not** expose it on a public interface — the port is
unauthenticated.

### Smoke check

```bash
curl -s http://127.0.0.1:8120/health
curl -s -X POST http://127.0.0.1:8120/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from XTTS","language":"en","speaker":"Viktor Menelaos"}' \
  | head -c 12; echo
```

Expect `{"ok": true, ...}` from `/health` and `RIFF....WAVE` from the POST. Then
in Hermes WebUI: Settings → Extensions → enable *XTTS v2* → Settings → TTS Engine
→ *XTTS v2 (local studio voice)* → **Listen** on a message.

### Mock mode (no model, no GPU)

```bash
python3 xtts_daemon.py --port 8120 --mock
```

`--mock` answers the identical HTTP contract with a short tone, which is how the
WebUI wiring and this repo's contract test are verified without weights.

### As a systemd user service

```bash
mkdir -p ~/.config/systemd/user
cp xtts-daemon.service ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/xtts-daemon.service   # set paths/venv/device
systemctl --user daemon-reload
systemctl --user enable --now xtts-daemon.service
systemctl --user status xtts-daemon.service
journalctl --user -u xtts-daemon.service -f
```

## HTTP contract

```http
GET /health
-> 200 {"ok":true,"mock":false,"model":"tts_models/multilingual/multi-dataset/xtts_v2",
        "device":"cpu","loaded":true,"speakers":["Ana Florence", ...]}

OPTIONS /
-> 204  Access-Control-Allow-Origin: <request Origin>
        Access-Control-Allow-Methods: POST, OPTIONS
        Access-Control-Allow-Headers: Content-Type
        Access-Control-Max-Age: 600
        Vary: Origin

POST /            Content-Type: application/json
{"text": "...", "language": "ru", "speaker": "Viktor Menelaos"}
-> 200 audio/wav  (RIFF/WAVE, 24 kHz, 16-bit mono, no-store)
-> 400  empty text / unknown speaker / invalid JSON
-> 415  non-JSON Content-Type
-> 413  body over 1 MiB or text over 20000 chars
-> 500  model/backend failure (never a silent 200)
```

`language` is one of the 17 XTTS v2 codes (`en, es, fr, de, it, pt, pl, tr, ru,
nl, cs, ar, zh-cn, ja, hu, ko, hi`); anything else falls back to `en`. `speaker`
must be one of the model's built-in speakers — `/health` lists the first ones,
and a 400 names the valid choices.

Long text is split on sentence boundaries (`<= 220` chars per chunk, hard-wrapped
if needed), synthesized chunk by chunk under a single lock, and concatenated into
one WAV, so the client always gets a single playable body.

### Why the CORS headers are mandatory

The WebUI page and this daemon are different origins, so the browser sends a
preflight and refuses the POST unless the daemon answers with
`Access-Control-Allow-Origin` (plus `-Methods`/`-Headers`). A daemon without them
shows up in the WebUI as *“XTTS daemon unreachable … does it allow CORS from this
page?”*. The reference daemon echoes the caller's `Origin` and sends
`Vary: Origin`; `*` is equally acceptable because the port, not the Origin
header, is the trust boundary.

## Test

```bash
python3 test_contract.py
```

It starts the daemon in `--mock` mode on an ephemeral port and asserts the
health payload, the CORS preflight, a RIFF/WAVE POST at 24 kHz, sentence
splitting of long input, and the 400/415 rejections. No model, no network.

## Notes and limitations

- The model is warmed in a background thread at startup, so `/health` answers
  immediately and a synthesis request that arrives mid-load waits on the model
  lock instead of failing. On a cold cache the download (~1.8 GB) can take
  minutes; warm the daemon once after install before pressing **Listen**, since
  the extension gives up after 120 s.
- One request at a time per model: synthesis is serialized; concurrent requests
  queue.
- No streaming: the reply is fully synthesized before the first byte goes out.
- Requests longer than the client's deadline (120 s in the extension) are
  dropped by the browser, not by the daemon; use `--device cuda` or shorter
  replies for CPU-only hosts.
- XTTS v2 model weights are licensed under the Coqui Public Model License
  (non-commercial). This daemon's own code is MIT.
- Voice cloning (`speaker_wav`) is out of scope here: the contract carries a
  built-in speaker name only.

## License

MIT — see [LICENSE](LICENSE). The XTTS v2 model and its weights are covered by
the Coqui Public Model License and are not distributed here.
