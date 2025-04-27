# ROOT/EXAMPLES/SEARCHAGENT/SRC/SAMSARA/SAMSARA.PY
"""
SamsaraAgent – stateful wrapper around Fireworks model + Tavily enrichment
"""

import os
import re
import json
import logging
from collections import deque
from dotenv import load_dotenv
from typing import Dict, List

from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)
from fastapi.middleware.cors import CORSMiddleware

from examples.search_agent.src.samsara.providers.model_provider import ModelProvider
from examples.search_agent.src.samsara.providers.search_provider import SearchProvider

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ───────────────────────────── heuristics ─────────────────────────────
_PERIOD_RE = re.compile(
    r"(?:\b\d{4}\b)|"
    r"(first|last)\s+day|"
    r"(high\s*school|middle\s*school|college|university)|"
    r"(when\s+i\s+was\b)|"
    r"(when\s+i(?:'| a)?m\b)|"
    r"(age\s+\d{1,2})",
    re.I,
)


def looks_like_period(text: str) -> bool:
    return bool(_PERIOD_RE.search(text))


def extract_urls(text: str) -> list[str]:
    return re.findall(r"(https?://\S+)", text)


# ───────────────────────────── agent ─────────────────────────────
class SamsaraAgent(AbstractAgent):
    _HISTORY_WINDOW = 18  # messages (user + assistant) before summarising

    def __init__(self, name: str = "Samsara"):
        super().__init__(name)

        model_key = os.getenv("MODEL_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not model_key:
            raise ValueError("MODEL_API_KEY missing in .env")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY missing in .env")

        self.model = ModelProvider(model_key)
        self.search = SearchProvider(tavily_key)

        # per-session state
        self.conversations: Dict[str, deque[Dict[str, str]]] = {}
        self.summaries: Dict[str, str] = {}  # sid -> rolling summary

    # ───────────────── private helpers ─────────────────
    async def _summarise(self, sid: str) -> None:
        """
        Condense history to keep the context window small.
        """
        history = list(self.conversations[sid])
        prompt = (
            "Condense the following Samsara chat into ≤120 words focusing on:\n"
            "• period label the user gave\n"
            "• memories / emotions they shared\n"
            "• their personality at that time\n"
            "• any artifacts provided.\n\n"
            "CHAT:\n"
            + json.dumps(history, ensure_ascii=False)
        )
        summary = await self.model.query([{"role": "user", "content": prompt}])
        summary = summary.strip()

        # replace conversation with a single system summary message
        self.conversations[sid] = deque(
            [{"role": "system", "content": f"[CONTEXT SUMMARY] {summary}"}],
            maxlen=512,
        )
        self.summaries[sid] = summary

    async def _filter_irrelevant(self, sid: str, prompt: str) -> bool:
        """
        Improved relevance filter. Returns True if the prompt is on-topic.

        Heuristic:
        1. If the user is giving a new period, always relevant.
        2. Otherwise ask the model to classify, but supply the running
           summary (or recent turns) as context so it understands what
           ‘on-topic’ means. This dramatically reduces false negatives.
        """
        # direct period spec ⇒ always relevant
        if looks_like_period(prompt):
            return True

        # Build minimal context (summary if we have one, else last 6 messages)
        context = self.summaries.get(sid)
        if context is None:
            recent = list(self.conversations.get(sid, []))[-6:]
            context = "\n".join(msg["content"] for msg in recent)

        classifier_prompt = (
            "You are a binary classifier. Return **ONLY** the single word "
            "'RELATED' or 'UNRELATED'.\n\n"
            "Conversation context (for reference, do not quote back):\n"
            f"{context}\n\n"
            "Guidelines:\n"
            "• Mark as RELATED if the message continues or deepens the role-play, "
            "shares memories or feelings from that time, asks questions "
            "to that self, or changes the target time period.\n"
            "• Mark as UNRELATED if the user veers off into an entirely "
            "different subject (e.g. asking for cooking recipes, current events, "
            "jokes, programming advice, random trivia) **and** does not "
            "mention the chosen time period or the self they are speaking to.\n\n"
            f"User message:\n{prompt}"
        )

        verdict = await self.model.query([{"role": "user", "content": classifier_prompt}])
        return verdict.strip().upper().startswith("RELATED")

    # ──────────────────── main entrypoint ────────────────────
    async def assist(
        self, session: Session, query: Query, rh: ResponseHandler
    ):
        sid = f"{session.processor_id}:{session.activity_id}"
        history = self.conversations.setdefault(sid, deque(maxlen=512))

        # append user turn
        history.append({"role": "user", "content": query.prompt})

        # 0) onboarding --------------------------------------------------
        if len(history) == 1 and not looks_like_period(query.prompt):
            greet = (
                "👋 Hey, I’m Samsara – I become you at any point in your life.\n\n"
                "Tell me the point in your life you want me to become. For example:\n"
                " • First day of high school after moving\n"
                " • When I’m 30 and living in Tokyo\n"
                " • A specific date you remember vividly\n\n"
                "I’ll chat as that version of you until you say otherwise. 👀"
            )
            stream = rh.create_text_stream("FINAL_RESPONSE")
            await stream.emit_chunk(greet)
            await stream.complete()
            history.append({"role": "assistant", "content": greet})
            await rh.complete()
            return

        # 1) relevance gate ---------------------------------------------
        if not await self._filter_irrelevant(sid, query.prompt):
            msg = (
                "I’m here solely to talk as you in your chosen time period. "
                "Ask me about that version of you, tell me a new period, or give me more relevant information!"
            )
            stream = rh.create_text_stream("FINAL_RESPONSE")
            await stream.emit_chunk(msg)
            await stream.complete()
            history.append({"role": "assistant", "content": msg})
            await rh.complete()
            return

        # 2) artifact ingestion -----------------------------------------
        for url in extract_urls(query.prompt):
            try:
                extracted = await self.search.extract([url])
                artifact = f"[ARTIFACT from {url}]\n{json.dumps(extracted, indent=2)}"
                history.append({"role": "assistant", "content": artifact})

                astream = rh.create_text_stream("LINK_INFO")
                await astream.emit_chunk(artifact)
                await astream.complete()
            except Exception as e:
                logger.warning("Tavily error %s -> %s", url, e)
                err = f"⚠️ Couldn’t fetch {url}"
                err_stream = rh.create_text_stream("LINK_INFO_ERROR")
                await err_stream.emit_chunk(err)
                await err_stream.complete()

        # 3) model streaming response -----------------------------------
        tstream = rh.create_text_stream("FINAL_RESPONSE")
        assistant_chunks: List[str] = []

        async for chunk in self.model.query_stream(list(history)):
            await tstream.emit_chunk(chunk)
            assistant_chunks.append(chunk)

        await tstream.complete()

        assistant_full = "".join(assistant_chunks)
        history.append({"role": "assistant", "content": assistant_full})

        # 4) summarise if needed ----------------------------------------
        if len(history) > self._HISTORY_WINDOW:
            await self._summarise(sid)

        await rh.complete()


# ─────────────────────────── FastAPI glue ────────────────────────────
if __name__ == "__main__":
    agent = SamsaraAgent()
    server = DefaultServer(agent)

    server._app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server.run()
