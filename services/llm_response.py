import logging
import os
import random
import re
from difflib import SequenceMatcher

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)

LLM_MODEL_CANDIDATES = ["gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"]
LLM_MAX_OUTPUT_TOKENS = 90
LLM_TEMPERATURE = 0.6
LLM_MAX_REPLY_LENGTH = 280
RECENT_REPLIES_LIMIT = 5
REPLY_SIMILARITY_THRESHOLD = 0.88

GENERIC_REPLIES = {
    "интересная мысль.",
    "хорошее наблюдение.",
    "да, в этом есть смысл.",
    "тут ты уловил важный момент.",
    "интересная мысль",
    "хорошее наблюдение",
    "да в этом есть смысл",
    "тут ты уловил важный момент",
}

ADDRESSING_RULES = """
Ты обязан отразить конкретную мысль ученика.
Нельзя отвечать общей фразой, которую можно вставить к любому ответу.
Если ученик написал "он наивный", ответ должен продолжать именно эту мысль.
Если ученик написал "отец всё решает", ответ должен опираться именно на это.
Если ученик написал "ему страшно", ответ должен прямо развивать именно эту мысль.
Не ограничивайся пустым согласием.
""".strip()

STRONG_SYSTEM_PROMPT = f"""
Ты — живой, спокойный и внимательный преподаватель литературы.
Ты помогаешь подростку понять «Капитанскую дочку».
Отвечай коротко, естественно и по-человечески.
Ты не экзаменатор и не учебник.
Ответ: 1–3 короткие фразы.
Без канцелярита.
Без нового вопроса.
Без длинного пересказа.

{ADDRESSING_RULES}
""".strip()

REGEN_INSTRUCTIONS = """
Ответ слишком общий.
Скажи конкретно, на какую мысль ученика ты отвечаешь.
Не начинай с пустой универсальной фразы.
""".strip()

UNIVERSAL_FALLBACK_REPLIES = [
    "Ничего, давай держаться за саму ситуацию героя.",
    "Это нормально, мы как раз разбираем всё по шагам.",
    "Можно пока опереться на то, что происходило в этом фрагменте.",
    "Не страшно, мысль может прийти по ходу.",
    "Давай попробуем оттолкнуться от самого поступка героя.",
]

DONT_KNOW_REPLIES = [
    "Ничего, давай держаться за саму ситуацию героя.",
    "Это нормально, мы как раз разбираем всё по шагам.",
    "Можно пока опереться на то, что происходило в этом фрагменте.",
    "Не страшно, мысль может прийти по ходу.",
    "Давай попробуем оттолкнуться от самого поступка героя.",
]

DONT_KNOW_PATTERNS = {
    "не знаю",
    "не знаю.",
    "не понял",
    "не поняла",
    "хз",
    "хз.",
    "не очень понял",
    "без понятия",
    "сложно",
    "не уверен",
    "не уверена",
    "не очень понимаю",
    "не понимаю",
    "затрудняюсь",
}

MEANINGLESS_PATTERNS = {
    ".",
    "..",
    "...",
    "-",
    "--",
    "—",
    "ээ",
    "эм",
    "мм",
    "а",
    "ы",
}

_client: AsyncOpenAI | None = None


def sanitize_user_answer(user_answer: str) -> str:
    return " ".join(user_answer.strip().split())


def normalize_user_answer(user_answer: str) -> str:
    return sanitize_user_answer(user_answer).lower()


def _compact_text(text: str) -> str:
    return re.sub(r"[^\w\s-]+", "", text, flags=re.UNICODE)


def is_empty_or_meaningless_answer(user_answer: str) -> bool:
    sanitized = sanitize_user_answer(user_answer)
    if not sanitized:
        return True

    normalized = normalize_user_answer(sanitized)
    if normalized in MEANINGLESS_PATTERNS:
        return True

    compact = _compact_text(normalized)
    return not compact


def should_use_llm(user_answer: str) -> bool:
    if is_empty_or_meaningless_answer(user_answer):
        return False

    normalized = normalize_user_answer(user_answer)
    compact = _compact_text(normalized)
    return normalized not in DONT_KNOW_PATTERNS and compact not in DONT_KNOW_PATTERNS


def detect_answer_type(user_answer: str) -> str:
    if not should_use_llm(user_answer):
        return "dont_know"
    return "normal"


def get_dont_know_reply() -> str:
    return random.choice(DONT_KNOW_REPLIES)


def get_universal_fallback_reply() -> str:
    return random.choice(UNIVERSAL_FALLBACK_REPLIES)


def trim_recent_replies(recent_replies: list[str] | None) -> list[str]:
    if not recent_replies:
        return []
    return recent_replies[-RECENT_REPLIES_LIMIT:]


def normalize_reply_text(reply: str) -> str:
    return _compact_text(reply.lower()).strip()


def is_generic_reply(reply: str) -> bool:
    normalized = normalize_reply_text(reply)
    if not normalized:
        return True
    return normalized in {normalize_reply_text(item) for item in GENERIC_REPLIES}


def is_similar_to_recent(reply: str, recent_replies: list[str] | None) -> bool:
    if not recent_replies:
        return False

    normalized_reply = normalize_reply_text(reply)
    for previous in recent_replies:
        similarity = SequenceMatcher(
            None, normalized_reply, normalize_reply_text(previous)
        ).ratio()
        if similarity >= REPLY_SIMILARITY_THRESHOLD:
            return True
    return False


def _shorten_reply(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""

    sentences: list[str] = []
    current = ""
    for char in cleaned:
        current += char
        if char in ".!?":
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    shortened = " ".join(sentences[:3]) if sentences else cleaned
    if len(shortened) <= LLM_MAX_REPLY_LENGTH:
        return shortened

    cut = shortened[:LLM_MAX_REPLY_LENGTH].rstrip()
    last_punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_punct > 40:
        return cut[: last_punct + 1].strip()
    return cut.rstrip(",;:- ") + "."


def _get_client() -> AsyncOpenAI | None:
    global _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[DEBUG] OPENAI_API_KEY not found")
        return None

    if _client is None:
        _client = AsyncOpenAI(api_key=api_key)

    return _client


def _build_user_prompt(
    block_title: str, block_text: str, question: str, user_answer: str
) -> str:
    return (
        f"Текущий раздел: {block_title}\n\n"
        f"Контекст:\n{block_text}\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Ответ ученика:\n{user_answer}\n\n"
        "Среагируй именно на мысль ученика, а не вообще на тему.\n"
        "Нельзя отвечать универсальной фразой.\n"
        "Нужно коротко показать, что ты понял, что именно он сказал, и чуть развить эту мысль."
    )


def _get_system_prompt(answer_type: str) -> str:
    return STRONG_SYSTEM_PROMPT


async def _call_model(
    *,
    answer_type: str,
    block_title: str,
    block_text: str,
    question: str,
    user_answer: str,
    extra_instructions: str = "",
) -> str | None:
    client = _get_client()
    if client is None:
        return None

    instructions = _get_system_prompt(answer_type)
    instructions = STRONG_SYSTEM_PROMPT
    if extra_instructions:
        instructions = f"{instructions}\n\n{extra_instructions}"

    for model_name in LLM_MODEL_CANDIDATES:
        try:
            logger.info("[DEBUG] trying_model=%s", model_name)
            response = await client.responses.create(
                model=model_name,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                temperature=LLM_TEMPERATURE,
                instructions=instructions,
                input=_build_user_prompt(
                    block_title=block_title,
                    block_text=block_text,
                    question=question,
                    user_answer=user_answer,
                ),
            )
            return _shorten_reply(response.output_text or "")
        except Exception:
            logger.exception("[DEBUG] OpenAI generation failed for model=%s", model_name)

    return None


async def generate_llm_reply(
    block_title: str,
    block_text: str,
    question: str,
    user_answer: str,
    recent_replies: list[str] | None = None,
) -> tuple[str, bool, bool, str]:
    sanitized_answer = sanitize_user_answer(user_answer)
    should_llm = should_use_llm(sanitized_answer)
    answer_type = detect_answer_type(sanitized_answer)
    recent_replies = trim_recent_replies(recent_replies)

    logger.info('[DEBUG] user_answer="%s"', sanitized_answer)
    logger.info("[DEBUG] answer_type=%s", answer_type)
    logger.info("[DEBUG] should_use_llm=%s", should_llm)

    if not should_llm:
        return get_dont_know_reply(), False, False, answer_type

    reply = await _call_model(
        answer_type=answer_type,
        block_title=block_title,
        block_text=block_text,
        question=question,
        user_answer=sanitized_answer,
    )

    if reply and not is_generic_reply(reply) and not is_similar_to_recent(reply, recent_replies):
        return reply, True, True, answer_type

    logger.info("[DEBUG] first_llm_reply_invalid=%s", True)
    reply = await _call_model(
        answer_type=answer_type,
        block_title=block_title,
        block_text=block_text,
        question=question,
        user_answer=sanitized_answer,
        extra_instructions=REGEN_INSTRUCTIONS,
    )

    if reply and not is_generic_reply(reply) and not is_similar_to_recent(reply, recent_replies):
        return reply, True, True, answer_type

    return get_universal_fallback_reply(), True, False, answer_type
