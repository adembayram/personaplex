import asyncio
import logging
from aiohttp import web
import aiohttp
import torch
from .models import loaders, LMGen
import sentencepiece
import sphn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SESSIONS = 5


class BaseState:
    """
    LMModel weights and tokenizer are shared between sessions,
    totally read-only data is stored here. Mimi instances are not here —
    each session creates its own mimi instance.
    """
    def __init__(self, lm, text_tokenizer, device, sample_rate, frame_rate):
        self.lm = lm                        # LMModel — only weights, stateless
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate



class SessionContext:
    """
    Each WebSocket connection has its own independent inference context.
    mimi, other_mimi and lm_gen are session-specific and do not mix.
    """
    def __init__(self, ws, base: BaseState):
        self.ws = ws
        self.base = base
        self.session_id = id(ws)
        self.queue: asyncio.Queue = asyncio.Queue()

        
        self.mimi = loaders.get_mimi(loaders.DEFAULT_REPO, base.device)
        self.other_mimi = loaders.get_mimi(loaders.DEFAULT_REPO, base.device)

       
        self.lm_gen = LMGen(
            base.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=base.device,
            frame_rate=self.mimi.frame_rate,
        )

        # Opus stream I/O
        self.opus_reader = sphn.OpusStreamReader(base.sample_rate)
        self.opus_writer = sphn.OpusStreamWriter(base.sample_rate)

    def cleanup(self):
        """Free GPU memory when session is closed."""
        try:
            del self.mimi
            del self.other_mimi
            del self.lm_gen
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"[session {self.session_id}] cleanup error: {e}")



async def session_inference_worker(session: SessionContext):
    """
    Each session has its own worker.
    Processes incoming pcm chunks in order.
    """
    base = session.base

    try:
        while True:
            pcm_chunk = await session.queue.get()

            if pcm_chunk is None:           # Poison pill → close
                session.queue.task_done()
                break

            if session.ws.closed:
                session.queue.task_done()
                continue

            try:
                with torch.no_grad():
                    chunk = torch.from_numpy(pcm_chunk)
                    chunk = chunk.to(device=base.device)[None, None]

                    codes = session.mimi.encode(chunk)
                    _ = session.other_mimi.encode(chunk)

                    for c in range(codes.shape[-1]):
                        tokens = session.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue

                        main_pcm = session.mimi.decode(tokens[:, 1:9]).cpu()
                        session.opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = (
                                base.text_tokenizer
                                .id_to_piece(text_token)
                                .replace("▁", " ")
                            )
                            if not session.ws.closed:
                                await session.ws.send_bytes(
                                    b"\x02" + bytes(_text, encoding="utf8")
                                )
            except Exception as e:
                logger.error(
                    f"[session {session.session_id}] inference error: {e}"
                )
            finally:
                session.queue.task_done()

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[session {session.session_id}] worker crashed: {e}")


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_chat(request: web.Request):
    # Check active session count
    active_sessions: dict = request.app["active_sessions"]

    if len(active_sessions) >= MAX_SESSIONS:
        logger.warning("Maximum session count reached, connection rejected.")
        return web.Response(status=503, text="Server full, please wait.")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    base: BaseState = request.app["base_state"]
    session = SessionContext(ws, base)
    active_sessions[session.session_id] = session

    logger.info(
        f"[session {session.session_id}] connected. "
        f"Active sessions: {len(active_sessions)}"
    )

    # Start inference worker for this session
    worker_task = asyncio.create_task(session_inference_worker(session))

    async def recv_loop():
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                data = message.data
                if len(data) > 1 and data[0] == 1:      # kind=1 → audio
                    session.opus_reader.append_bytes(data[1:])
                    pcm = session.opus_reader.read_pcm()
                    if pcm is not None and pcm.shape[-1] > 0:
                        await session.queue.put(pcm)
            elif message.type in (
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
            ):
                break

    async def send_loop():
        try:
            while not ws.closed:
                await asyncio.sleep(0.005)
                out_bytes = session.opus_writer.read_bytes()
                if len(out_bytes) > 0:
                    await ws.send_bytes(b"\x01" + out_bytes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[session {session.session_id}] send error: {e}")

    recv_task = asyncio.create_task(recv_loop())
    send_task = asyncio.create_task(send_loop())

    try:
        await asyncio.wait(
            [recv_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        # Stop recv and send tasks
        for task in [recv_task, send_task]:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop worker with poison pill
        await session.queue.put(None)
        try:
            await asyncio.wait_for(worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            worker_task.cancel()

        # Session cleanup and free GPU memory
        active_sessions.pop(session.session_id, None)
        session.cleanup()

        if not ws.closed:
            await ws.close()

        logger.info(
            f"[session {session.session_id}] closed. "
            f"Active sessions: {len(active_sessions)}"
        )

    return ws


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info("Loading models...")

    # Temporary mimi — only read sample_rate and frame_rate values
    _tmp_mimi = loaders.get_mimi(loaders.DEFAULT_REPO, device)
    sample_rate = _tmp_mimi.sample_rate
    frame_rate  = _tmp_mimi.frame_rate
    del _tmp_mimi
    torch.cuda.empty_cache()

    text_tokenizer = sentencepiece.SentencePieceProcessor(
        loaders.get_tokenizer(loaders.DEFAULT_REPO)
    )
    lm = loaders.get_moshi_lm(loaders.DEFAULT_REPO, device=device)
    lm.eval()

    base_state = BaseState(
        lm=lm,
        text_tokenizer=text_tokenizer,
        device=device,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
    )

    logger.info("Models ready. Server starting...")

    app = web.Application()
    app["base_state"] = base_state
    app["active_sessions"] = {}         # session_id → SessionContext

    app.router.add_get("/api/chat", handle_chat)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
