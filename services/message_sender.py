import asyncio
import logging
import random
import re
from typing import Callable, Optional

from telegram import Bot
from telegram.constants import ChatAction


_bot: Optional[Bot] = None
logger = logging.getLogger(__name__)


def configure_bot(bot: Bot) -> None:
    global _bot
    _bot = bot


def split_text_to_sentences(text: str) -> list[str]:
    chunks: list[str] = []

    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue

        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks.extend(part.strip() for part in parts if part.strip())

    return chunks


async def send_typing_message(
    chat_id: int,
    text: str,
    reply_markup=None,
    flow_guard: Callable[[], bool] | None = None,
):
    if _bot is None:
        raise RuntimeError("Bot is not configured.")

    if flow_guard and not flow_guard():
        logger.info("[DEBUG] skipped outdated message")
        return False

    try:
        await _bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(random.uniform(1.0, 2.0))
    except asyncio.CancelledError:
        logger.info("[DEBUG] typing cancelled")
        raise

    if flow_guard and not flow_guard():
        logger.info("[DEBUG] skipped outdated message")
        return False

    try:
        await _bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
        await asyncio.sleep(random.uniform(0.5, 1.5))
    except asyncio.CancelledError:
        logger.info("[DEBUG] typing cancelled")
        raise

    return True
