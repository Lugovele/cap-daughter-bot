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

GUIDED_SYSTEM_PROMPT = 'Ты — живой, спокойный и доброжелательный преподаватель литературы.\nТы помогаешь подростку понять «Капитанскую дочку» через вопросы.\nТвоя задача — отреагировать на конкретный ответ ученика и помочь ему думать дальше.\n\nГлавное ограничение:\n- оценивай ответ пользователя только относительно текущего показанного фрагмента\n- не используй знания о произведении за пределами этого фрагмента\n- не намекай на будущие события и не требуй выводов, для которых в shown_text ещё нет основания\n- если материала для глубокого вывода мало, сужай задачу до простой детали из shown_text\n\nПравила:\n- отвечай по-русски\n- 1–3 короткие фразы\n- если мысль верная внутри текущего фрагмента, коротко поддержи и добавь один смысловой акцент\n- если мысль частичная, мягко уточни, какой детали из shown_text не хватает\n- если ответ слабый или "не знаю", сузь задачу, подсвети одну заметную деталь из shown_text и мягко подтолкни к следующей попытке\n- не повторяй вопрос дословно\n- не используй слова "неправильно", "ошибка", "неверно"\n- не говори "смотри на сцену", "в этом эпизоде", "в разделе"\n- не ссылайся на номера, меню или структуру курса\n- не отвечай универсальной заготовкой\n- звучишь как живой наставник, а не как система'

GUIDED_REGEN_INSTRUCTIONS = 'Предыдущий ответ получился слишком общим.\nОтветь конкретно на мысль ученика и на смысл вопроса.\nОпирайся только на shown_text и allowed_points.\nЕсли ученик затрудняется, дай одну живую подсказку без ссылки на сцену, эпизод или раздел.'

GUIDED_FALLBACK_REPLIES = [
    'Попробуй оттолкнуться от того, что сейчас чувствует или выбирает герой.',
    'Давай сузим: здесь важно понять, что для героя становится главным в этот момент.',
    'Можно начать проще: что меняется для героя и почему это для него важно?',
    'Подумай о поступке героя: что он показывает о его характере?',
]

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
    return not is_empty_or_meaningless_answer(user_answer)


def is_dont_know_answer(user_answer: str) -> bool:
    normalized = normalize_user_answer(user_answer)
    compact = _compact_text(normalized)
    return normalized in DONT_KNOW_PATTERNS or compact in DONT_KNOW_PATTERNS


def detect_answer_type(user_answer: str) -> str:
    if is_empty_or_meaningless_answer(user_answer) or is_dont_know_answer(user_answer):
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


def get_guided_fallback_reply() -> str:
    return random.choice(GUIDED_FALLBACK_REPLIES)


def _format_guided_options(options: list[str] | None) -> str:
    if not options:
        return 'не указаны'
    return "\n".join(f"{index}. {option}" for index, option in enumerate(options, start=1))


def _format_guided_points(points: list[str] | str | None) -> str:
    if not points:
        return "не указаны"
    if isinstance(points, str):
        return points.strip() or "не указаны"
    return "\n".join(f"- {point}" for point in points if str(point).strip()) or "не указаны"


def _build_guided_user_prompt(
    *,
    context_title: str,
    context_text: str,
    question: str,
    user_answer: str,
    correct_direction: str = "",
    learning_goal: str = "",
    allowed_points: list[str] | str | None = None,
    forbidden_future_context: list[str] | str | None = None,
    answer_options: list[str] | None = None,
    selected_answer: str | None = None,
    is_correct: bool | None = None,
    answer_type: str = "normal",
) -> str:
    status = 'затрудняется' if answer_type == "dont_know" else 'не указано'
    if is_correct is True:
        status = 'по сути верный'
    elif is_correct is False:
        status = 'неточный или неполный'

    return (
        f"Тема фрагмента: {context_title}\n\n"
        f"shown_text — текст, который уже видел пользователь:\n{context_text}\n\n"
        f"Вопрос ученику:\n{question}\n\n"
        f"Учебная цель вопроса:\n{learning_goal or correct_direction or 'помочь заметить смысл, который уже есть в shown_text'}\n\n"
        f"Допустимые смысловые опоры только из shown_text:\n{_format_guided_points(allowed_points or correct_direction)}\n\n"
        f"Запрещённый будущий контекст:\n{_format_guided_points(forbidden_future_context or 'не использовать события, детали и выводы за пределами shown_text')}\n\n"
        f"Ответ ученика:\n{user_answer}\n\n"
        f"Варианты ответа, если они есть:\n{_format_guided_options(answer_options)}\n\n"
        f"Выбранный вариант, если он есть:\n{selected_answer or 'не указано'}\n\n"
        f"Оценка ответа системой:\n{status}\n\n"
        f"Правильное смысловое направление в границах shown_text:\n{correct_direction or 'помочь увидеть главный смысл уже показанной ситуации'}\n\n"
        'Сформулируй короткую живую реакцию. '
        'Не выходи за пределы shown_text. '
        'Если ученик затрудняется, не повторяй вопрос, а сузь задачу, дай одну смысловую опору и подтолкни к следующей попытке. '
        'Если ответ неточный, мягко направь к одной детали, которая прямо есть в shown_text. '
        'Если ответ хороший, коротко поддержи и не перегружай разбором.'
    )


async def _call_guided_model(
    *,
    context_title: str,
    context_text: str,
    question: str,
    user_answer: str,
    correct_direction: str = "",
    learning_goal: str = "",
    allowed_points: list[str] | str | None = None,
    forbidden_future_context: list[str] | str | None = None,
    answer_options: list[str] | None = None,
    selected_answer: str | None = None,
    is_correct: bool | None = None,
    answer_type: str = "normal",
    extra_instructions: str = "",
) -> str | None:
    client = _get_client()
    if client is None:
        return None

    instructions = GUIDED_SYSTEM_PROMPT
    if extra_instructions:
        instructions = f"{instructions}\n\n{extra_instructions}"

    user_prompt = _build_guided_user_prompt(
        context_title=context_title,
        context_text=context_text,
        question=question,
        user_answer=user_answer,
        correct_direction=correct_direction,
        learning_goal=learning_goal,
        allowed_points=allowed_points,
        forbidden_future_context=forbidden_future_context,
        answer_options=answer_options,
        selected_answer=selected_answer,
        is_correct=is_correct,
        answer_type=answer_type,
    )

    for model_name in LLM_MODEL_CANDIDATES:
        try:
            logger.info("[DEBUG] guided_trying_model=%s", model_name)
            response = await client.responses.create(
                model=model_name,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                temperature=LLM_TEMPERATURE,
                instructions=instructions,
                input=user_prompt,
            )
            return _shorten_reply(response.output_text or "")
        except Exception:
            logger.exception("[DEBUG] guided OpenAI generation failed for model=%s", model_name)

    return None


async def generate_guided_reply(
    *,
    context_title: str,
    context_text: str,
    question: str,
    user_answer: str,
    correct_direction: str = "",
    learning_goal: str = "",
    allowed_points: list[str] | str | None = None,
    forbidden_future_context: list[str] | str | None = None,
    answer_options: list[str] | None = None,
    selected_answer: str | None = None,
    is_correct: bool | None = None,
    recent_replies: list[str] | None = None,
) -> tuple[str, bool, bool, str]:
    sanitized_answer = sanitize_user_answer(user_answer)
    answer_type = detect_answer_type(sanitized_answer)
    should_llm = should_use_llm(sanitized_answer)
    recent_replies = trim_recent_replies(recent_replies)

    logger.info('[DEBUG] guided_user_answer="%s"', sanitized_answer)
    logger.info("[DEBUG] guided_answer_type=%s", answer_type)
    logger.info("[DEBUG] guided_should_use_llm=%s", should_llm)

    if not should_llm:
        return get_guided_fallback_reply(), False, False, answer_type

    reply = await _call_guided_model(
        context_title=context_title,
        context_text=context_text,
        question=question,
        user_answer=sanitized_answer,
        correct_direction=correct_direction,
        learning_goal=learning_goal,
        allowed_points=allowed_points,
        forbidden_future_context=forbidden_future_context,
        answer_options=answer_options,
        selected_answer=selected_answer,
        is_correct=is_correct,
        answer_type=answer_type,
    )
    if reply and not is_generic_reply(reply) and not is_similar_to_recent(reply, recent_replies):
        return reply, True, True, answer_type

    reply = await _call_guided_model(
        context_title=context_title,
        context_text=context_text,
        question=question,
        user_answer=sanitized_answer,
        correct_direction=correct_direction,
        learning_goal=learning_goal,
        allowed_points=allowed_points,
        forbidden_future_context=forbidden_future_context,
        answer_options=answer_options,
        selected_answer=selected_answer,
        is_correct=is_correct,
        answer_type=answer_type,
        extra_instructions=GUIDED_REGEN_INSTRUCTIONS,
    )
    if reply and not is_generic_reply(reply) and not is_similar_to_recent(reply, recent_replies):
        return reply, True, True, answer_type

    return get_guided_fallback_reply(), should_llm, False, answer_type

STAGE3_SYSTEM_PROMPT = """
Ты — внимательный, живой и доброжелательный преподаватель литературы.
Ты помогаешь подростку понять «Капитанскую дочку» через подробный адаптированный пересказ, близкий к оригиналу.
Твоя задача — дать содержательную реакцию именно на ответ ученика по текущему эпизоду.

Главное ограничение:
- оценивай ответ только по тексту эпизода, который уже показан пользователю
- не используй знания о следующих событиях и об оригинале за пределами этого текста
- не требуй от ученика вывода, который нельзя сделать из уже показанного фрагмента

Правила реакции:
- отвечай по-русски
- 2–4 коротких предложения
- сначала признай то, что ученик уловил или попытался уловить
- затем мягко развей, уточни или верни мысль к сути сцены
- опирайся на конкретный ответ ученика и на данный эпизод
- не отвечай универсальной фразой, которую можно вставить к любому ответу
- не повторяй вопрос
- не используй слова: "неправильно", "ошибка", "неверно"
- не будь сухим экзаменатором
- не пересказывай весь эпизод заново
- звучишь как спокойный умный преподаватель, который ведёт рядом
""".strip()

STAGE3_REGEN_INSTRUCTIONS = """
Предыдущая реакция получилась слишком общей или неудачной.
Скажи конкретно, на какую мысль ученика ты отвечаешь.
Мягко свяжи ответ ученика с ключевым смыслом эпизода.
Не используй слова "неправильно", "ошибка", "неверно".
""".strip()

STAGE3_FORBIDDEN_WORDS = {"неправильно", "ошибка", "неверно"}
STAGE3_GENERIC_REPLIES = {
    "интересная мысль",
    "интересная мысль.",
    "хорошее наблюдение",
    "хорошее наблюдение.",
    "да, в этом есть смысл",
    "да, в этом есть смысл.",
    "тут ты уловил важный момент",
    "тут ты уловил важный момент.",
}
STAGE3_MAX_OUTPUT_TOKENS = 180
STAGE3_MAX_REPLY_LENGTH = 520


def _format_stage3_list(items: list[str] | None) -> str:
    if not items:
        return "не указано"
    return "\n".join(f"- {item}" for item in items)


def _shorten_stage3_feedback(text: str) -> str:
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

    shortened = " ".join(sentences[:4]) if sentences else cleaned
    if len(shortened) <= STAGE3_MAX_REPLY_LENGTH:
        return shortened

    cut = shortened[:STAGE3_MAX_REPLY_LENGTH].rstrip()
    last_punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_punct > 80:
        return cut[: last_punct + 1].strip()
    return cut.rstrip(",;:- ") + "."


def _contains_stage3_forbidden_words(reply: str) -> bool:
    normalized = normalize_user_answer(reply)
    return any(word in normalized for word in STAGE3_FORBIDDEN_WORDS)


def _build_stage3_user_prompt(
    *,
    episode_text: str,
    question: str,
    user_answer: str,
    core_meanings: list[str] | None,
    good_answer_signals: list[str] | None,
    common_mistakes: list[str] | None,
    reaction_style: str | None,
) -> str:
    return (
        f"shown_text — текст, который уже видел пользователь:\n{episode_text}\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Ответ пользователя:\n{user_answer}\n\n"
        f"Ключевые смыслы, допустимые только из shown_text:\n{_format_stage3_list(core_meanings)}\n\n"
        f"Признаки хорошего ответа:\n{_format_stage3_list(good_answer_signals)}\n\n"
        f"Типичные ошибки понимания:\n{_format_stage3_list(common_mistakes)}\n\n"
        f"Стиль реакции:\n{reaction_style or 'поддерживающий, спокойный, содержательный'}\n\n"
        "Сформулируй короткую, живую, педагогически точную реакцию на ответ пользователя. "
        "Нужно сначала показать, что ты понял его мысль, затем мягко развить или уточнить её. "
        "Если ответ поверхностный или уходит в сторону, спокойно верни к тому, что прямо видно в shown_text. "
        "Не выходи за пределы shown_text и не намекай на будущие события. "
        "Не повторяй вопрос и не отвечай шаблонно."
    )


def _build_stage3_fallback_feedback(
    *,
    user_answer: str,
    core_meanings: list[str] | None,
) -> str:
    first_meaning = (core_meanings or ["важно увидеть внутренний смысл ситуации"])[0]
    answer = sanitize_user_answer(user_answer)
    if answer:
        return _shorten_stage3_feedback(
            f"Я вижу, от какой мысли ты отталкиваешься: {answer}. "
            f"Здесь стоит связать это с главным смыслом происходящего: {first_meaning}. "
            "Попробуй держать в центре не только событие, но и то, что оно говорит о характере героя."
        )
    return _shorten_stage3_feedback(
        f"Можно начать с самого главного: {first_meaning}. "
        "Здесь важны не только события, но и внутренняя опора, которую герой получает перед взрослой жизнью."
    )


async def generate_stage3_feedback(
    *,
    episode_text: str,
    question: str,
    user_answer: str,
    core_meanings: list[str] | None = None,
    good_answer_signals: list[str] | None = None,
    common_mistakes: list[str] | None = None,
    reaction_style: str | None = None,
) -> tuple[str, bool]:
    sanitized_answer = sanitize_user_answer(user_answer)
    client = _get_client()
    if client is None:
        logger.warning("[DEBUG] stage3 AI feedback fallback: OPENAI_API_KEY missing")
        return _build_stage3_fallback_feedback(
            user_answer=sanitized_answer,
            core_meanings=core_meanings,
        ), False

    user_prompt = _build_stage3_user_prompt(
        episode_text=episode_text,
        question=question,
        user_answer=sanitized_answer,
        core_meanings=core_meanings,
        good_answer_signals=good_answer_signals,
        common_mistakes=common_mistakes,
        reaction_style=reaction_style,
    )

    for attempt in range(2):
        instructions = STAGE3_SYSTEM_PROMPT
        if attempt == 1:
            instructions = f"{instructions}\n\n{STAGE3_REGEN_INSTRUCTIONS}"

        for model_name in LLM_MODEL_CANDIDATES:
            try:
                logger.info("[DEBUG] stage3 trying_model=%s attempt=%s", model_name, attempt + 1)
                response = await client.responses.create(
                    model=model_name,
                    max_output_tokens=STAGE3_MAX_OUTPUT_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    instructions=instructions,
                    input=user_prompt,
                )
                reply = _shorten_stage3_feedback(response.output_text or "")
                if reply and not is_generic_reply(reply) and normalize_user_answer(reply) not in STAGE3_GENERIC_REPLIES and not _contains_stage3_forbidden_words(reply):
                    return reply, True
            except Exception:
                logger.exception("[DEBUG] Stage 3 OpenAI generation failed for model=%s", model_name)

    logger.warning("[DEBUG] stage3 AI feedback fallback: generation failed")
    return _build_stage3_fallback_feedback(
        user_answer=sanitized_answer,
        core_meanings=core_meanings,
    ), False
