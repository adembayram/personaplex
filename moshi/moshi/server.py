import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]

MAX_SESSIONS = 5


# ---------------------------------------------------------------------------
# Helper functions (original code)
# ---------------------------------------------------------------------------

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ---------------------------------------------------------------------------
# Shared (read-only) base state
# ---------------------------------------------------------------------------

class BaseState:
    """
    LM weights and tokenizer are shared between sessions,
    totally read-only data. mimi_weight path is also stored here —
    each session uses this path when creating a new mimi.
    """
    def __init__(
        self,
        lm: LMModel,
        mimi_weight: str,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        device: torch.device,
        voice_prompt_dir: Optional[str],
        sample_rate: int,
        frame_rate: float,
        frame_size: int,
    ):
        self.lm = lm
        self.mimi_weight = mimi_weight          # local file path
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frame_size = frame_size


# ---------------------------------------------------------------------------
# Session-level completely isolated state
# ---------------------------------------------------------------------------

class SessionContext:
    """
    Each WebSocket connection has its own independent inference context.
    mimi, other_mimi and lm_gen are session-specific and do not mix.
    """
    def __init__(self, ws: web.WebSocketResponse, base: BaseState):
        self.ws = ws
        self.base = base
        self.session_id = id(ws)
        self.clog = ColorizedLog.randomize()

        # Each session creates its own mimi instances from local path
        self.mimi: MimiModel = loaders.get_mimi(base.mimi_weight, base.device)
        self.other_mimi: MimiModel = loaders.get_mimi(base.mimi_weight, base.device)

        # LMGen session-specific (KV cache and audio buffer stateful)
        self.lm_gen = LMGen(
            base.lm,
            audio_silence_frame_cnt=int(0.5 * base.frame_rate),
            sample_rate=base.sample_rate,
            device=base.device,
            frame_rate=base.frame_rate,
        )

        # Start streaming mode (original code)
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Opus stream I/O
        self.opus_reader = sphn.OpusStreamReader(base.sample_rate)
        self.opus_writer = sphn.OpusStreamWriter(base.sample_rate)

    def reset_streaming(self):
        """Reset all states before speech starts."""
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def cleanup(self):
        """Free GPU memory when session is closed."""
        try:
            del self.mimi
            del self.other_mimi
            del self.lm_gen
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[session {self.session_id}] cleanup error: {e}")


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_chat(request: web.Request):
    active_sessions: dict = request.app["active_sessions"]

    if len(active_sessions) >= MAX_SESSIONS:
        logger.warning("Maximum session count reached, connection rejected.")
        return web.Response(status=503, text="Server is full, please try again later.")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    base: BaseState = request.app["base_state"]
    session = SessionContext(ws, base)
    active_sessions[session.session_id] = session
    clog = session.clog

    peer = request.remote
    peer_port = request.transport.get_extra_info("peername")[1]
    clog.log("info", f"[{session.session_id}] connected {peer}:{peer_port} | "
                     f"Active sessions: {len(active_sessions)}")

    # --- Set voice and text prompt (original code) ---
    requested_voice_prompt_path = None
    voice_prompt_path = None
    if base.voice_prompt_dir is not None:
        voice_prompt_filename = request.query.get("voice_prompt", "")
        if voice_prompt_filename:
            requested_voice_prompt_path = os.path.join(
                base.voice_prompt_dir, voice_prompt_filename
            )
        if requested_voice_prompt_path and not os.path.exists(requested_voice_prompt_path):
            await ws.close()
            active_sessions.pop(session.session_id, None)
            session.cleanup()
            raise FileNotFoundError(
                f"Voice prompt '{voice_prompt_filename}' not found: '{base.voice_prompt_dir}'"
            )
        voice_prompt_path = requested_voice_prompt_path

    if voice_prompt_path and session.lm_gen.voice_prompt != voice_prompt_path:
        if voice_prompt_path.endswith(".pt"):
            session.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        else:
            session.lm_gen.load_voice_prompt(voice_prompt_path)

    text_prompt = request.query.get("text_prompt", "")
    session.lm_gen.text_prompt_tokens = (
        base.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
        if text_prompt else None
    )

    seed = int(request.query["seed"]) if "seed" in request.query else None
    if seed is not None and seed != -1:
        seed_all(seed)

    # Reset streaming state
    session.reset_streaming()

    close = False

    async def recv_loop():
        nonlocal close
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    clog.log("error", f"{ws.exception()}")
                    break
                elif message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                ):
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    clog.log("error", f"unexpected message type {message.type}")
                    continue
                data = message.data
                if not isinstance(data, bytes) or len(data) == 0:
                    continue
                kind = data[0]
                if kind == 1:  # audio
                    session.opus_reader.append_bytes(data[1:])
                else:
                    clog.log("warning", f"unknown kind {kind}")
        finally:
            close = True
            clog.log("info", f"[{session.session_id}] recv_loop closed")

    async def opus_loop():
        all_pcm_data = None
        while True:
            if close:
                return
            await asyncio.sleep(0.001)
            pcm = session.opus_reader.read_pcm()
            if pcm is None or pcm.shape[-1] == 0:
                continue
            all_pcm_data = pcm if all_pcm_data is None else np.concatenate((all_pcm_data, pcm))

            while all_pcm_data.shape[-1] >= base.frame_size:
                chunk = all_pcm_data[: base.frame_size]
                all_pcm_data = all_pcm_data[base.frame_size:]
                chunk = torch.from_numpy(chunk).to(device=base.device)[None, None]

                with torch.no_grad():
                    codes = session.mimi.encode(chunk)
                    _ = session.other_mimi.encode(chunk)

                    for c in range(codes.shape[-1]):
                        tokens = session.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        main_pcm = session.mimi.decode(tokens[:, 1:9])
                        _ = session.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        session.opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = base.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                            if not ws.closed:
                                await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))

    async def send_loop():
        while True:
            if close:
                return
            await asyncio.sleep(0.001)
            msg = session.opus_writer.read_bytes()
            if len(msg) > 0:
                await ws.send_bytes(b"\x01" + msg)

    # Send handshake
    await ws.send_bytes(b"\x00")
    clog.log("info", f"[{session.session_id}] handshake sent")

    tasks = [
        asyncio.create_task(recv_loop()),
        asyncio.create_task(opus_loop()),
        asyncio.create_task(send_loop()),
    ]

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        active_sessions.pop(session.session_id, None)
        session.cleanup()
        if not ws.closed:
            await ws.close()
        clog.log("info", f"[{session.session_id}] session closed | "
                         f"Aktif session: {len(active_sessions)}")

    return ws


# ---------------------------------------------------------------------------
# Helper: voice and static path (original code)
# ---------------------------------------------------------------------------

def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    logger.info("voice prompts downloading")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} -> {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz does not contain 'voices/' directory")
    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("static content downloading")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true")
    parser.add_argument("--gradio-tunnel-token", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl", type=str)
    parser.add_argument("--max-sessions", type=int, default=MAX_SESSIONS)
    args = parser.parse_args()

    global MAX_SESSIONS
    MAX_SESSIONS = args.max_sessions

    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt_dir:
        assert os.path.exists(args.voice_prompt_dir), f"Directory not found: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path not found: {static_path}"
    logger.info(f"static_path = {static_path}")

    args.device = torch_auto_device(args.device)
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            logger.error("gradio not installed: pip install gradio")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token or secrets.token_urlsafe(32)

    # Download model weights (local paths)
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)

    # Create temporary mimi for frame_size, then delete
    _tmp = loaders.get_mimi(args.mimi_weight, args.device)
    sample_rate = _tmp.sample_rate
    frame_rate  = _tmp.frame_rate
    frame_size  = int(_tmp.sample_rate / _tmp.frame_rate)
    del _tmp
    torch.cuda.empty_cache()
    logger.info("mimi metadata retrieved")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")

    base_state = BaseState(
        lm=lm,
        mimi_weight=args.mimi_weight,   # lokal path — session'lar bunu kullanacak
        text_tokenizer=text_tokenizer,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        frame_size=frame_size,
    )

    app = web.Application()
    app["base_state"] = base_state
    app["active_sessions"] = {}

    app.router.add_get("/api/chat", handle_chat)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        logger.info(f"serving static content: {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Web UI: {protocol}://{host_ip}:{args.port}")

    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None)
        logger.info(f"Tunnel: {tunnel}")

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()