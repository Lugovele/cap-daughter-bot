import asyncio
import logging


from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import ContextTypes

from services.llm_response import (
    detect_answer_type,
    generate_guided_reply,
    generate_stage3_feedback,
    sanitize_user_answer,
    trim_recent_replies,
)
from services.message_sender import send_typing_message, split_text_to_sentences
from services.progress import get_user_progress, save_stage2_progress, save_stage3_progress, save_user_progress


logger = logging.getLogger(__name__)

MENU_BUTTON = "📍 Меню"
STAGE_1 = "stage_1"
STAGE_2 = "stage_2"
STAGE_3 = "stage_3"
DEFAULT_STAGE = STAGE_1
STAGE_TITLES = {
    STAGE_1: "1. Герои и сюжет",
    STAGE_2: "2. Пересказ",
    STAGE_3: "3. С цитатами",
}
STAGE_PLACEHOLDERS = {}
RECENT_REPLIES_KEY = "recent_bot_replies"
FLOW_ID_KEY = "flow_id"
FLOW_TASK_KEY = "flow_task"
DONT_KNOW_COUNT_KEY = "dont_know_count"
STOP_COMMANDS = {"стоп", "stop"}
ATTEMPT_COUNT_KEY = "attempt_count"
QUESTION_STATUS_KEY = "question_status"
QUESTION_STATUS_WAITING = "waiting_user_answer"
QUESTION_STATUS_AI_FOLLOWUP = "ai_followup_sent"
QUESTION_STATUS_COMPLETED = "completed"
MAX_QUESTION_ATTEMPTS = 2
CURRENT_STAGE_KEY = "current_stage"
CURRENT_BLOCK_KEY = "current_block"
QUESTION_INDEX_KEY = "question_index"
AWAITING_ANSWER_KEY = "awaiting_answer"
STAGE2_CHAPTER_KEY = "stage2_chapter_id"
STAGE2_QUESTION_INDEX_KEY = "stage2_question_index"
STAGE2_MODE_KEY = "stage2_mode"
STAGE2_MODE_CHAPTERS = "chapters"
STAGE2_MODE_SCENE = "scene"
STAGE2_MODE_AWAITING_ANSWER = "awaiting_answer"
STAGE2_MODE_FINISHED_CHAPTER = "finished_chapter"

CALLBACK_MENU_CONTINUE = "menu_continue"
CALLBACK_MENU_HELP = "help"
CALLBACK_MENU_OPEN = "menu_open"
CALLBACK_MENU_ORIGINAL = "original"
STAGE_CALLBACK_PREFIX = "select_"
BLOCK_CALLBACK_PREFIX = "jump_"
STAGE2_CHAPTER_PREFIX = "stage2_chapter_"
CALLBACK_STAGE2_NEXT = "stage2_next"
CALLBACK_STAGE2_CHAPTERS = "stage2_chapters"
CALLBACK_STAGE2_CONTINUE = "stage2_continue"
STAGE3_EPISODE_KEY = "stage3_episode_id"
STAGE3_STEP_KEY = "stage3_step"
STAGE3_MODE_KEY = "stage3_mode"
STAGE3_MODE_EPISODES = "episodes"
STAGE3_MODE_TEXT = "text"
STAGE3_MODE_OPEN_QUESTION = "open_question"
STAGE3_MODE_FEEDBACK = "feedback"
STAGE3_MODE_POST_FEEDBACK_NAVIGATION = "post_feedback_navigation"
STAGE3_MODE_AWAITING_ANSWER = STAGE3_MODE_OPEN_QUESTION
STAGE3_MODE_FINISHED_EPISODE = STAGE3_MODE_POST_FEEDBACK_NAVIGATION
STAGE3_EPISODE_PREFIX = "stage3_episode_"
CALLBACK_STAGE3_NEXT = "stage3_next"
CALLBACK_STAGE3_EPISODES = "stage3_episodes"


def get_block_map(context: ContextTypes.DEFAULT_TYPE) -> dict[str, dict]:
    course = context.application.bot_data["course"]
    return {block["id"]: block for block in course["blocks"]}


def get_blocks(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    return context.application.bot_data["course"]["blocks"]


def get_block_title(block: dict) -> str:
    return block.get("title") or block["id"]


def get_stage2_content(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.application.bot_data["stage_2"]


def get_stage2_chapters(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    stage_data = get_stage2_content(context)
    return stage_data.get("episodes") or stage_data.get("chapters", [])


def get_stage2_chapter_map(context: ContextTypes.DEFAULT_TYPE) -> dict[str, dict]:
    return {
        get_stage2_chapter_id(chapter): chapter
        for chapter in get_stage2_chapters(context)
    }


def get_stage2_chapter_id(chapter: dict) -> str:
    return str(chapter.get("key") or chapter.get("id"))


def get_stage2_chapter_label(chapter: dict) -> str:
    return str(chapter.get("menu_label") or chapter.get("title") or get_stage2_chapter_id(chapter))


def get_stage3_content(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.application.bot_data["stage_3"]


def get_stage3_episodes(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    return get_stage3_content(context).get("episodes", [])


def get_stage3_episode_id(episode: dict) -> str:
    return str(episode.get("key") or episode.get("id"))


def get_stage3_episode_label(episode: dict) -> str:
    return str(episode.get("menu_label") or episode.get("title") or get_stage3_episode_id(episode))


def get_stage3_episode_map(context: ContextTypes.DEFAULT_TYPE) -> dict[str, dict]:
    return {
        get_stage3_episode_id(episode): episode
        for episode in get_stage3_episodes(context)
    }


def format_stage2_question(question: str, options: list[str] | None = None) -> str:
    question = str(question or "").strip()
    if not options:
        return question

    option_lines = [
        f"{index}. {option}"
        for index, option in enumerate(options, start=1)
    ]
    return f"{question}\n\n" + "\n".join(option_lines)


def build_stage2_step(block_type: str, block: dict) -> dict:
    return {
        "type": block_type,
        "prompt": block.get("prompt", ""),
        "options": block.get("options", []),
        "correct": block.get("correct"),
        "correct_reaction": block.get("correct_reaction", "Да, именно так."),
        "correct_explanation": block.get("correct_explanation", ""),
        "learning_goal": block.get("learning_goal", ""),
        "allowed_points": block.get("allowed_points", []),
        "forbidden_future_context": block.get("forbidden_future_context", []),
        "wrong_responses": block.get("wrong_responses", {}),
        "wrong_reaction": block.get("wrong_reaction", "Пока не совсем точно."),
        "wrong_hint": block.get("wrong_hint", "Попробуй посмотреть на ситуацию шире."),
        "retry_prompt": block.get("retry_prompt", "Выбери один ответ ещё раз."),
        "reaction_map": block.get("reaction_map", {}),
    }


def get_stage2_steps(chapter: dict) -> list[dict]:
    steps = []
    if chapter.get("meaning_block"):
        steps.append(build_stage2_step("meaning", chapter["meaning_block"]))
    if chapter.get("line_block"):
        steps.append(build_stage2_step("line", chapter["line_block"]))
    if chapter.get("personal_block"):
        steps.append(build_stage2_step("personal", chapter["personal_block"]))
    return steps


def get_stage2_questions(chapter: dict) -> list[dict]:
    steps = get_stage2_steps(chapter)
    if steps:
        return [
            {
                "q": format_stage2_question(step["prompt"], step["options"]),
                "hint": step.get("correct_explanation") or "",
            }
            for step in steps
        ]

    legacy_questions = chapter.get("questions")
    if legacy_questions:
        return [
            {
                "q": get_stage2_question_text(question_obj),
                "hint": get_stage2_question_hint(chapter, index),
            }
            for index, question_obj in enumerate(legacy_questions)
        ]

    questions = []
    if chapter.get("meaning_question"):
        questions.append(
            {
                "q": format_stage2_question(
                    chapter.get("meaning_question"),
                    chapter.get("meaning_options"),
                ),
                "hint": str(chapter.get("meaning_explanation", "")).strip(),
            }
        )
    if chapter.get("line_question"):
        questions.append(
            {
                "q": format_stage2_question(
                    chapter.get("line_question"),
                    chapter.get("line_options"),
                ),
                "hint": str(chapter.get("line_explanation", "")).strip(),
            }
        )
    if chapter.get("personal_choice_question"):
        questions.append(
            {
                "q": format_stage2_question(
                    chapter.get("personal_choice_question"),
                    chapter.get("personal_choice_options"),
                ),
                "hint": str(chapter.get("personal_choice_reaction", "")).strip(),
            }
        )
    return questions


def parse_stage2_option_answer(user_answer: str, options: list[str]) -> int | None:
    normalized = sanitize_user_answer(user_answer).lower()
    if not normalized:
        return None

    if normalized.isdigit():
        option_index = int(normalized) - 1
        if 0 <= option_index < len(options):
            return option_index

    for index, option in enumerate(options):
        option_text = str(option).strip().lower()
        if normalized == option_text:
            return index
        if normalized in {f"{index + 1}. {option_text}", f"{index + 1}) {option_text}"}:
            return index

    return None


def get_stage2_question_text(question_obj) -> str:
    if isinstance(question_obj, dict):
        return str(question_obj.get("q", "")).strip()
    return str(question_obj).strip()


def get_stage2_question_hint(chapter: dict, question_index: int) -> str:
    questions = chapter.get("questions", [])
    if 0 <= question_index < len(questions):
        question_obj = questions[question_index]
        if isinstance(question_obj, dict):
            return str(question_obj.get("hint", "")).strip()

    legacy_hints = chapter.get("hints", [])
    if 0 <= question_index < len(legacy_hints):
        return str(legacy_hints[question_index]).strip()

    return ""


def as_text_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def build_fragment_context(
    *,
    shown_text: str,
    question: str,
    learning_goal: str = "",
    allowed_points=None,
    forbidden_future_context=None,
) -> dict:
    return {
        "shown_text": str(shown_text or "").strip(),
        "question": str(question or "").strip(),
        "learning_goal": str(learning_goal or "").strip(),
        "allowed_points": as_text_list(allowed_points),
        "forbidden_future_context": as_text_list(forbidden_future_context)
        or [
            "Не использовать события, детали и выводы, которых нет в текущем показанном фрагменте."
        ],
    }


def get_stage1_fragment_context(block: dict, question: str) -> dict:
    learning_goal = block.get("learning_goal") or block.get("hint") or ""
    allowed_points = block.get("allowed_points") or block.get("hint") or block.get("text", "")
    return build_fragment_context(
        shown_text=block.get("shown_text") or block.get("text", ""),
        question=question,
        learning_goal=learning_goal,
        allowed_points=allowed_points,
        forbidden_future_context=block.get("forbidden_future_context"),
    )


def get_stage2_fragment_context(chapter: dict, step: dict) -> dict:
    correct_direction = get_stage2_correct_direction(step)
    allowed_points = step.get("allowed_points") or [
        point
        for point in [
            correct_direction,
            step.get("correct_reaction"),
            step.get("correct_explanation"),
        ]
        if point
    ]
    return build_fragment_context(
        shown_text=chapter.get("shown_text")
        or chapter.get("text")
        or chapter.get("summary")
        or chapter.get("short_summary")
        or "",
        question=step.get("prompt", ""),
        learning_goal=step.get("learning_goal") or correct_direction,
        allowed_points=allowed_points,
        forbidden_future_context=step.get("forbidden_future_context")
        or chapter.get("forbidden_future_context"),
    )


def get_stage3_fragment_context(episode: dict) -> dict:
    return build_fragment_context(
        shown_text=episode.get("shown_text") or episode.get("text", ""),
        question=episode.get("question", ""),
        learning_goal=episode.get("learning_goal")
        or "Помочь понять смысл вопроса только по уже показанному тексту.",
        allowed_points=episode.get("allowed_points")
        or episode.get("core_meanings")
        or episode.get("good_answer_signals"),
        forbidden_future_context=episode.get("forbidden_future_context"),
    )


def is_stop_command(text: str) -> bool:
    return sanitize_user_answer(text).lower() in STOP_COMMANDS


def build_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[MENU_BUTTON]],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def build_menu_inline_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("▶️ Продолжить", callback_data=CALLBACK_MENU_CONTINUE)],
        [InlineKeyboardButton("ℹ️ Как пользоваться", callback_data=CALLBACK_MENU_HELP)],
        *[
            [
                InlineKeyboardButton(
                    title,
                    callback_data=f"{STAGE_CALLBACK_PREFIX}{stage_id}",
                )
            ]
            for stage_id, title in STAGE_TITLES.items()
        ],
        [InlineKeyboardButton("📖 Оригинал книги", callback_data=CALLBACK_MENU_ORIGINAL)],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_menu_only_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Меню", callback_data=CALLBACK_MENU_OPEN)]]
    )


def build_stage_1_contents_inline_keyboard(
    context: ContextTypes.DEFAULT_TYPE,
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                get_block_title(block),
                callback_data=f"{BLOCK_CALLBACK_PREFIX}{block['id']}",
            )
        ]
        for block in get_blocks(context)
    ]
    return InlineKeyboardMarkup(keyboard)


def build_stage2_chapters_inline_keyboard(
    context: ContextTypes.DEFAULT_TYPE,
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                get_stage2_chapter_label(chapter),
                callback_data=f"{STAGE2_CHAPTER_PREFIX}{get_stage2_chapter_id(chapter)}",
            )
        ]
        for chapter in get_stage2_chapters(context)
    ]
    return InlineKeyboardMarkup(keyboard)


def build_stage2_finished_inline_keyboard(
    has_next_chapter: bool,
) -> InlineKeyboardMarkup:
    keyboard = []
    if has_next_chapter:
        keyboard.append([InlineKeyboardButton("Продолжить", callback_data=CALLBACK_STAGE2_NEXT)])
    keyboard.append([InlineKeyboardButton("Назад к эпизодам", callback_data=CALLBACK_STAGE2_CHAPTERS)])
    return InlineKeyboardMarkup(keyboard)


def build_stage2_continue_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Продолжить", callback_data=CALLBACK_STAGE2_CONTINUE)]]
    )


def build_stage3_episodes_inline_keyboard(
    context: ContextTypes.DEFAULT_TYPE,
) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(
                get_stage3_episode_label(episode),
                callback_data=f"{STAGE3_EPISODE_PREFIX}{get_stage3_episode_id(episode)}",
            )
        ]
        for episode in get_stage3_episodes(context)
    ]
    keyboard.append([InlineKeyboardButton("Меню", callback_data=CALLBACK_MENU_OPEN)])
    return InlineKeyboardMarkup(keyboard)


def build_stage3_finished_inline_keyboard(
    has_next_episode: bool,
) -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Продолжить", callback_data=CALLBACK_STAGE3_NEXT)],
        [InlineKeyboardButton("Назад к эпизодам", callback_data=CALLBACK_STAGE3_EPISODES)],
    ]
    return InlineKeyboardMarkup(keyboard)


def set_contents_mode(context: ContextTypes.DEFAULT_TYPE, is_active: bool) -> None:
    context.user_data["awaiting_contents_choice"] = is_active


def is_contents_mode(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get("awaiting_contents_choice", False))


def reset_recent_replies(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data[RECENT_REPLIES_KEY] = []


def reset_dont_know_count(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data[DONT_KNOW_COUNT_KEY] = 0


def increase_dont_know_count(context: ContextTypes.DEFAULT_TYPE) -> int:
    count = int(context.user_data.get(DONT_KNOW_COUNT_KEY, 0)) + 1
    context.user_data[DONT_KNOW_COUNT_KEY] = count
    return count


def reset_question_flow(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data[ATTEMPT_COUNT_KEY] = 0
    context.user_data[QUESTION_STATUS_KEY] = QUESTION_STATUS_WAITING


def increase_attempt_count(context: ContextTypes.DEFAULT_TYPE, user_id: int | None = None) -> int:
    if user_id is None:
        current_count = int(context.user_data.get(ATTEMPT_COUNT_KEY, 0))
    else:
        current_count = get_attempt_count(context, user_id)
    count = current_count + 1
    context.user_data[ATTEMPT_COUNT_KEY] = count
    return count


def get_attempt_count(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> int:
    progress = get_user_progress(user_id) or {}
    return int(context.user_data.get(ATTEMPT_COUNT_KEY, progress.get("attempt_count", 0) or 0))


def set_question_status(context: ContextTypes.DEFAULT_TYPE, status: str) -> None:
    context.user_data[QUESTION_STATUS_KEY] = status


def get_question_status(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> str:
    progress = get_user_progress(user_id) or {}
    return context.user_data.get(
        QUESTION_STATUS_KEY,
        progress.get("question_status", QUESTION_STATUS_WAITING),
    )


def is_substantial_open_answer(user_answer: str, answer_type: str) -> bool:
    normalized = sanitize_user_answer(user_answer).lower()
    words = [word for word in normalized.split() if any(char.isalpha() for char in word)]
    letters = [char for char in normalized if char.isalpha()]
    if answer_type == "dont_know":
        return True
    if words:
        return True
    return len(letters) >= 3


def answer_uses_allowed_points(user_answer: str, allowed_points=None) -> bool:
    points = as_text_list(allowed_points)
    if not points:
        return True

    answer_words = {
        word
        for word in sanitize_user_answer(user_answer).lower().split()
        if len(word) >= 4
    }
    if not answer_words:
        return False

    for point in points:
        point_words = {
            word
            for word in sanitize_user_answer(point).lower().split()
            if len(word) >= 4
        }
        if answer_words & point_words:
            return True
    return False


def evaluate_answer_result(
    *,
    user_answer: str,
    answer_type: str,
    attempt_count: int,
    is_correct: bool | None = None,
    requires_correct: bool = False,
    allowed_points=None,
) -> dict:
    reached_attempt_limit = attempt_count >= MAX_QUESTION_ATTEMPTS
    if requires_correct:
        is_correct_enough = bool(is_correct) or reached_attempt_limit
    else:
        is_substantial = is_substantial_open_answer(user_answer, answer_type)
        is_correct_enough = (
            is_substantial
            and (
                answer_uses_allowed_points(user_answer, allowed_points)
                or bool(sanitize_user_answer(user_answer))
            )
        )

    advance_allowed = is_correct_enough
    return {
        "is_correct_enough": is_correct_enough,
        "needs_followup": not advance_allowed,
        "advance_allowed": advance_allowed,
        "reached_attempt_limit": reached_attempt_limit,
        "status": QUESTION_STATUS_COMPLETED
        if advance_allowed
        else QUESTION_STATUS_AI_FOLLOWUP,
    }


def set_current_stage(context: ContextTypes.DEFAULT_TYPE, stage: str) -> None:
    context.user_data[CURRENT_STAGE_KEY] = stage


def set_stage2_state(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chapter_id: str | None,
    question_index: int = 0,
    mode: str = STAGE2_MODE_CHAPTERS,
    attempt_count: int | None = None,
    question_status: str | None = None,
) -> None:
    status = question_status or QUESTION_STATUS_WAITING
    context.user_data[CURRENT_STAGE_KEY] = STAGE_2
    context.user_data["current_work"] = "captains_daughter"
    context.user_data["current_episode"] = chapter_id
    context.user_data["current_step"] = question_index
    context.user_data["current_block"] = mode
    context.user_data["current_question_type"] = mode
    context.user_data["current_attempt_state"] = status
    context.user_data[STAGE2_CHAPTER_KEY] = chapter_id
    context.user_data[STAGE2_QUESTION_INDEX_KEY] = question_index
    context.user_data[STAGE2_MODE_KEY] = mode
    context.user_data[QUESTION_STATUS_KEY] = status
    if attempt_count is not None:
        context.user_data[ATTEMPT_COUNT_KEY] = attempt_count


def get_stage2_state(
    context: ContextTypes.DEFAULT_TYPE, user_id: int
) -> tuple[str | None, int, str]:
    progress = get_user_progress(user_id) or {}
    chapter_id = context.user_data.get(
        STAGE2_CHAPTER_KEY,
        progress.get("current_chapter_id") or progress.get("current_episode"),
    )
    question_index = int(
        context.user_data.get(
            STAGE2_QUESTION_INDEX_KEY,
            progress.get("current_question_index", progress.get("current_step", 0)) or 0,
        )
    )
    mode = context.user_data.get(
        STAGE2_MODE_KEY,
        progress.get("current_mode")
        or progress.get("current_stage2_block")
        or STAGE2_MODE_CHAPTERS,
    )
    return chapter_id, question_index, mode


def set_stage3_state(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    episode_id: str | None,
    step: int = 0,
    mode: str = STAGE3_MODE_EPISODES,
    attempt_count: int | None = None,
    question_status: str | None = None,
) -> None:
    status = question_status or QUESTION_STATUS_WAITING
    context.user_data[CURRENT_STAGE_KEY] = STAGE_3
    context.user_data["current_work"] = "captains_daughter"
    context.user_data["current_episode"] = episode_id
    context.user_data["current_step"] = step
    context.user_data["current_block"] = mode
    context.user_data["current_question_type"] = mode
    context.user_data["current_attempt_state"] = status
    context.user_data[STAGE3_EPISODE_KEY] = episode_id
    context.user_data[STAGE3_STEP_KEY] = step
    context.user_data[STAGE3_MODE_KEY] = mode
    context.user_data[QUESTION_STATUS_KEY] = status
    if attempt_count is not None:
        context.user_data[ATTEMPT_COUNT_KEY] = attempt_count


def get_stage3_state(
    context: ContextTypes.DEFAULT_TYPE, user_id: int
) -> tuple[str | None, int, str]:
    progress = get_user_progress(user_id) or {}
    episode_id = context.user_data.get(
        STAGE3_EPISODE_KEY,
        progress.get("current_episode_id") or progress.get("current_episode"),
    )
    step = int(context.user_data.get(STAGE3_STEP_KEY, progress.get("current_step", 0) or 0))
    mode = context.user_data.get(
        STAGE3_MODE_KEY,
        progress.get("current_mode")
        or progress.get("current_stage3_block")
        or STAGE3_MODE_EPISODES,
    )
    return episode_id, step, mode


def get_current_stage(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> str:
    progress = get_user_progress(user_id) or {}
    return context.user_data.get(
        CURRENT_STAGE_KEY,
        progress.get("current_stage", DEFAULT_STAGE),
    )


def set_dialog_state(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    current_stage: str = DEFAULT_STAGE,
    current_block: str | None,
    question_index: int,
    awaiting_answer: bool,
    attempt_count: int | None = None,
    question_status: str | None = None,
) -> None:
    context.user_data[CURRENT_STAGE_KEY] = current_stage
    context.user_data[CURRENT_BLOCK_KEY] = current_block
    context.user_data[QUESTION_INDEX_KEY] = question_index
    context.user_data[AWAITING_ANSWER_KEY] = awaiting_answer
    context.user_data[QUESTION_STATUS_KEY] = question_status or QUESTION_STATUS_WAITING
    if attempt_count is not None:
        context.user_data[ATTEMPT_COUNT_KEY] = attempt_count


def get_dialog_state(
    context: ContextTypes.DEFAULT_TYPE, user_id: int
) -> tuple[str, str | None, int, bool]:
    progress = get_user_progress(user_id) or {}
    current_stage = context.user_data.get(
        CURRENT_STAGE_KEY,
        progress.get("current_stage", DEFAULT_STAGE),
    )
    current_block = context.user_data.get(CURRENT_BLOCK_KEY, progress.get("current_block"))
    question_index = int(
        context.user_data.get(QUESTION_INDEX_KEY, progress.get("question_index", 0) or 0)
    )
    awaiting_answer = bool(context.user_data.get(AWAITING_ANSWER_KEY, True))
    return current_stage, current_block, question_index, awaiting_answer


def begin_new_flow(context: ContextTypes.DEFAULT_TYPE) -> str:
    flow_id = uuid4().hex
    context.user_data[FLOW_ID_KEY] = flow_id
    logger.info("[DEBUG] flow_id updated")
    return flow_id


def get_flow_id(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get(FLOW_ID_KEY, "")


def is_flow_active(context: ContextTypes.DEFAULT_TYPE, flow_id: str) -> bool:
    return get_flow_id(context) == flow_id


def interrupt_flow(context: ContextTypes.DEFAULT_TYPE) -> str:
    flow_id = begin_new_flow(context)
    task = context.user_data.get(FLOW_TASK_KEY)
    if task and not task.done():
        task.cancel()
        logger.info("[DEBUG] flow interrupted by menu action")
    context.user_data[FLOW_TASK_KEY] = None
    return flow_id


def start_flow_task(context: ContextTypes.DEFAULT_TYPE, coroutine) -> asyncio.Task:
    task = context.application.create_task(coroutine)
    context.user_data[FLOW_TASK_KEY] = task
    return task


async def send_guarded_message(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    flow_id: str,
    text: str,
    reply_markup=None,
) -> bool:
    if not is_flow_active(context, flow_id):
        logger.info("[DEBUG] skipped outdated message")
        return False

    return await send_typing_message(
        chat_id,
        text,
        reply_markup=reply_markup or build_main_keyboard(),
        flow_guard=lambda: is_flow_active(context, flow_id),
    )


async def show_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("[DEBUG] menu opened")
    flow_id = get_flow_id(context)
    set_contents_mode(context, False)
    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        "Выбери, что сделать:",
        reply_markup=build_menu_inline_keyboard(),
    )


async def show_stage_1_contents_menu(
    chat_id: int, context: ContextTypes.DEFAULT_TYPE
) -> None:
    flow_id = get_flow_id(context)
    set_contents_mode(context, True)
    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        "Выбери главу:",
        reply_markup=build_stage_1_contents_inline_keyboard(context),
    )


async def show_stage2_chapters_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    flow_id = get_flow_id(context)
    set_contents_mode(context, True)
    stage2 = get_stage2_content(context)
    chapters = get_stage2_chapters(context)
    if not chapters:
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            "2. Пересказ пока не содержит эпизодов.",
        )
        return

    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        f"{stage2['title']}\n\n{stage2.get('description', stage2.get('episode_menu_title', 'Выбери эпизод:'))}",
        reply_markup=build_stage2_chapters_inline_keyboard(context),
    )


async def send_stage2_question(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chapter_id: str,
    question_index: int,
    flow_id: str,
) -> None:
    if not is_flow_active(context, flow_id):
        logger.info("[DEBUG] flow interrupted")
        return

    chapter = get_stage2_chapter_map(context)[chapter_id]
    questions = get_stage2_questions(chapter)
    if not questions:
        set_stage2_state(
            context,
            chapter_id=chapter_id,
            question_index=0,
            mode=STAGE2_MODE_FINISHED_CHAPTER,
            attempt_count=0,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        save_stage2_progress(
            user_id,
            chapter_id,
            0,
            STAGE2_MODE_FINISHED_CHAPTER,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        await send_stage2_finished_menu(chat_id, context, chapter_id, flow_id)
        return

    safe_index = min(max(question_index, 0), len(questions) - 1)
    set_stage2_state(
        context,
        chapter_id=chapter_id,
        question_index=safe_index,
        mode=STAGE2_MODE_AWAITING_ANSWER,
        attempt_count=0,
        question_status=QUESTION_STATUS_WAITING,
    )
    save_stage2_progress(
        user_id,
        chapter_id,
        safe_index,
        STAGE2_MODE_AWAITING_ANSWER,
        question_status=QUESTION_STATUS_WAITING,
    )
    question_text = get_stage2_question_text(questions[safe_index])
    if not question_text:
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            "Вопрос для этой главы пока не заполнен.",
        )
        return

    await send_guarded_message(chat_id, context, flow_id, question_text)


async def send_stage2_step(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chapter_id: str,
    step_index: int,
    flow_id: str,
    reset_attempts: bool = True,
) -> None:
    if not is_flow_active(context, flow_id):
        logger.info("[DEBUG] flow interrupted")
        return

    chapter = get_stage2_chapter_map(context)[chapter_id]
    steps = get_stage2_steps(chapter)
    if step_index >= len(steps):
        await send_stage2_takeaway(chat_id, context, user_id, chapter_id, flow_id)
        return

    step = steps[step_index]
    attempt_count = 0 if reset_attempts else get_attempt_count(context, user_id)
    question_status = (
        QUESTION_STATUS_WAITING
        if reset_attempts
        else get_question_status(context, user_id)
    )
    set_stage2_state(
        context,
        chapter_id=chapter_id,
        question_index=step_index,
        mode=STAGE2_MODE_AWAITING_ANSWER,
        attempt_count=attempt_count,
        question_status=question_status,
    )
    save_stage2_progress(
        user_id,
        chapter_id,
        step_index,
        STAGE2_MODE_AWAITING_ANSWER,
        attempt_count=attempt_count,
        question_status=question_status,
    )
    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        format_stage2_question(step["prompt"], step["options"]),
    )


async def send_stage2_takeaway(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chapter_id: str,
    flow_id: str,
) -> None:
    chapter = get_stage2_chapter_map(context)[chapter_id]
    raw_takeaway = chapter.get("scene_takeaway", "")
    if isinstance(raw_takeaway, dict):
        title = str(raw_takeaway.get("title", "")).strip()
        points = raw_takeaway.get("points", [])
        point_lines = [f"- {point}" for point in points]
        takeaway = "\n".join([title, *point_lines]).strip()
    else:
        takeaway = str(raw_takeaway).strip()

    if takeaway:
        sent = await send_guarded_message(chat_id, context, flow_id, takeaway)
        if not sent:
            return

    step_count = len(get_stage2_steps(chapter))
    set_stage2_state(
        context,
        chapter_id=chapter_id,
        question_index=step_count,
        mode=STAGE2_MODE_FINISHED_CHAPTER,
        attempt_count=0,
        question_status=QUESTION_STATUS_COMPLETED,
    )
    save_stage2_progress(
        user_id,
        chapter_id,
        step_count,
        STAGE2_MODE_FINISHED_CHAPTER,
        question_status=QUESTION_STATUS_COMPLETED,
    )
    await send_stage2_finished_menu(chat_id, context, chapter_id, flow_id)


async def send_stage2_chapter(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chapter_id: str,
    flow_id: str,
) -> None:
    chapter = get_stage2_chapter_map(context)[chapter_id]
    set_contents_mode(context, False)
    set_stage2_state(
        context,
        chapter_id=chapter_id,
        question_index=0,
        mode=STAGE2_MODE_SCENE,
        attempt_count=0,
        question_status=QUESTION_STATUS_WAITING,
    )
    save_stage2_progress(
        user_id,
        chapter_id,
        0,
        STAGE2_MODE_SCENE,
        question_status=QUESTION_STATUS_WAITING,
    )

    sent = await send_guarded_message(chat_id, context, flow_id, chapter["title"])
    if not sent:
        return

    episode_text = chapter.get("summary") or chapter.get("text") or ""
    if episode_text:
        sent = await send_guarded_message(chat_id, context, flow_id, episode_text)
        if not sent:
            return

    short_summary = chapter.get("short_summary") or []
    if short_summary:
        summary_text = "Коротко:\n" + "\n".join(f"- {item}" for item in short_summary)
        sent = await send_guarded_message(chat_id, context, flow_id, summary_text)
        if not sent:
            return

    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        "Когда будешь готов, нажми «Продолжить».",
        reply_markup=build_stage2_continue_inline_keyboard(),
    )


def get_next_stage2_chapter_id(context: ContextTypes.DEFAULT_TYPE, chapter_id: str) -> str | None:
    chapters = get_stage2_chapters(context)
    for index, chapter in enumerate(chapters):
        if get_stage2_chapter_id(chapter) == chapter_id:
            navigation = chapter.get("navigation") or {}
            next_episode = navigation.get("next_episode")
            if next_episode is not None:
                next_episode_id = str(next_episode)
                chapter_map = get_stage2_chapter_map(context)
                if next_episode_id in chapter_map:
                    return next_episode_id
                next_episode_key = f"stage2_episode_{next_episode_id}"
                if next_episode_key in chapter_map:
                    return next_episode_key

            if index + 1 < len(chapters):
                return get_stage2_chapter_id(chapters[index + 1])
    return None


async def send_stage2_finished_menu(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    chapter_id: str,
    flow_id: str,
) -> None:
    next_chapter_id = get_next_stage2_chapter_id(context, chapter_id)
    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        "Глава завершена. Что делаем дальше?",
        reply_markup=build_stage2_finished_inline_keyboard(next_chapter_id is not None),
    )


async def show_stage3_episodes_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    flow_id = get_flow_id(context)
    set_contents_mode(context, True)
    stage3 = get_stage3_content(context)
    episodes = get_stage3_episodes(context)
    if not episodes:
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            "3. С цитатами пока не содержит эпизодов.",
        )
        return

    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        f"{stage3['title']}\n\n{stage3.get('description', stage3.get('episode_menu_title', 'Выбери эпизод:'))}",
        reply_markup=build_stage3_episodes_inline_keyboard(context),
    )


async def send_stage3_open_question(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    episode_id: str,
    flow_id: str,
    reset_attempts: bool = True,
) -> None:
    episode = get_stage3_episode_map(context)[episode_id]
    attempt_count = 0 if reset_attempts else get_attempt_count(context, user_id)
    question_status = (
        QUESTION_STATUS_WAITING
        if reset_attempts
        else get_question_status(context, user_id)
    )
    set_stage3_state(
        context,
        episode_id=episode_id,
        step=1,
        mode=STAGE3_MODE_OPEN_QUESTION,
        attempt_count=attempt_count,
        question_status=question_status,
    )
    save_stage3_progress(
        user_id,
        episode_id,
        1,
        STAGE3_MODE_OPEN_QUESTION,
        attempt_count=attempt_count,
        question_status=question_status,
    )
    question_text = episode.get("question", "Как ты думаешь, что здесь важно?")
    await send_guarded_message(chat_id, context, flow_id, question_text)

async def send_stage3_episode(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    episode_id: str,
    flow_id: str,
) -> None:
    episode = get_stage3_episode_map(context)[episode_id]
    set_contents_mode(context, False)
    logger.info("[DEBUG] stage3 episode opened: %s", episode_id)
    set_stage3_state(
        context,
        episode_id=episode_id,
        step=0,
        mode=STAGE3_MODE_TEXT,
        attempt_count=0,
        question_status=QUESTION_STATUS_WAITING,
    )
    save_stage3_progress(
        user_id,
        episode_id,
        0,
        STAGE3_MODE_TEXT,
        question_status=QUESTION_STATUS_WAITING,
    )

    sent = await send_guarded_message(chat_id, context, flow_id, episode["title"])
    if not sent:
        return

    for part in split_text_to_sentences(episode.get("text", "")):
        sent = await send_guarded_message(chat_id, context, flow_id, part)
        if not sent:
            return

    await send_stage3_open_question(chat_id, context, user_id, episode_id, flow_id)


def get_next_stage3_episode_id(context: ContextTypes.DEFAULT_TYPE, episode_id: str) -> str | None:
    episodes = get_stage3_episodes(context)
    episode_map = get_stage3_episode_map(context)
    for index, episode in enumerate(episodes):
        if get_stage3_episode_id(episode) == episode_id:
            navigation = episode.get("navigation") or {}
            next_episode = navigation.get("next_episode")
            if next_episode is not None:
                next_episode_id = str(next_episode)
                if next_episode_id in episode_map:
                    return next_episode_id
                next_episode_key = f"stage3_episode_{next_episode_id}"
                if next_episode_key in episode_map:
                    return next_episode_key
            if index + 1 < len(episodes):
                return get_stage3_episode_id(episodes[index + 1])
    return None


async def send_stage3_finished_menu(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    episode_id: str,
    flow_id: str,
) -> None:
    episode = get_stage3_episode_map(context)[episode_id]
    takeaway = episode.get("takeaway")
    if takeaway:
        sent = await send_guarded_message(chat_id, context, flow_id, takeaway)
        if not sent:
            return

    set_stage3_state(
        context,
        episode_id=episode_id,
        step=3,
        mode=STAGE3_MODE_POST_FEEDBACK_NAVIGATION,
        attempt_count=0,
        question_status=QUESTION_STATUS_COMPLETED,
    )
    save_stage3_progress(
        user_id,
        episode_id,
        3,
        STAGE3_MODE_POST_FEEDBACK_NAVIGATION,
        question_status=QUESTION_STATUS_COMPLETED,
    )
    await send_guarded_message(
        chat_id,
        context,
        flow_id,
        "Эпизод завершён. Что делаем дальше?",
        reply_markup=build_stage3_finished_inline_keyboard(
            get_next_stage3_episode_id(context, episode_id) is not None
        ),
    )


async def send_current_question(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    block_id: str,
    question_index: int,
    flow_id: str,
    reset_attempts: bool = True,
    user_id: int | None = None,
) -> None:
    if not is_flow_active(context, flow_id):
        logger.info("[DEBUG] flow interrupted")
        return

    attempt_count = 0
    question_status = QUESTION_STATUS_WAITING
    if not reset_attempts and user_id is not None:
        attempt_count = get_attempt_count(context, user_id)
        question_status = get_question_status(context, user_id)

    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block=block_id,
        question_index=question_index,
        awaiting_answer=True,
        attempt_count=attempt_count,
        question_status=question_status,
    )
    block = get_block_map(context)[block_id]
    questions = block["questions"]
    safe_index = min(max(question_index, 0), len(questions) - 1)

    sent = await send_guarded_message(chat_id, context, flow_id, questions[safe_index])
    if not sent:
        return


async def send_block(
    chat_id: int, context: ContextTypes.DEFAULT_TYPE, block_id: str, flow_id: str
) -> None:
    set_contents_mode(context, False)
    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block=block_id,
        question_index=0,
        awaiting_answer=False,
    )
    block = get_block_map(context)[block_id]

    for sentence in split_text_to_sentences(block["text"]):
        if not is_flow_active(context, flow_id):
            logger.info("[DEBUG] flow interrupted")
            return

        sent = await send_guarded_message(chat_id, context, flow_id, sentence)
        if not sent:
            return

    await send_current_question(chat_id, context, block_id, 0, flow_id)


async def start_selected_block(
    chat_id: int, context: ContextTypes.DEFAULT_TYPE, user_id: int, block_id: str
) -> None:
    save_user_progress(user_id, block_id, 0, STAGE_1)
    set_contents_mode(context, False)
    reset_recent_replies(context)
    reset_dont_know_count(context)
    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block=block_id,
        question_index=0,
        awaiting_answer=False,
    )
    logger.info("[DEBUG] selected_block=%s", block_id)
    await send_block(chat_id, context, block_id, get_flow_id(context))


async def skip_stage1_question(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    block_id: str,
    question_index: int,
    flow_id: str,
) -> None:
    block = get_block_map(context).get(block_id)
    if not block:
        await send_guarded_message(chat_id, context, flow_id, "Ок, открою начало курса.")
        await send_block(chat_id, context, "intro", flow_id)
        return

    reset_dont_know_count(context)
    set_question_status(context, QUESTION_STATUS_COMPLETED)
    next_question_index = question_index + 1
    if next_question_index < len(block["questions"]):
        save_user_progress(
            user_id,
            block_id,
            next_question_index,
            STAGE_1,
            attempt_count=0,
            question_status=QUESTION_STATUS_WAITING,
        )
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block=block_id,
            question_index=next_question_index,
            awaiting_answer=True,
            attempt_count=0,
            question_status=QUESTION_STATUS_WAITING,
        )
        await send_current_question(
            chat_id,
            context,
            block_id,
            next_question_index,
            flow_id,
        )
        return

    next_block_id = block["next"]
    if not next_block_id:
        save_user_progress(
            user_id,
            None,
            0,
            STAGE_1,
            attempt_count=0,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block=None,
            question_index=0,
            awaiting_answer=False,
            attempt_count=0,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            "Это был последний блок. Если хочешь пройти всё заново, открой меню.",
        )
        return

    save_user_progress(
        user_id,
        next_block_id,
        0,
        STAGE_1,
        attempt_count=0,
        question_status=QUESTION_STATUS_WAITING,
    )
    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block=next_block_id,
        question_index=0,
        awaiting_answer=False,
        attempt_count=0,
        question_status=QUESTION_STATUS_WAITING,
    )
    await send_block(chat_id, context, next_block_id, flow_id)


async def skip_current_step(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    current_stage: str,
    block_id: str | None,
    question_index: int,
    flow_id: str,
) -> None:
    logger.info("[DEBUG] stop command detected")
    sent = await send_guarded_message(chat_id, context, flow_id, "Ок, пропускаем этот вопрос.")
    if not sent:
        return

    if current_stage == STAGE_2:
        chapter_id, stage2_step_index, mode = get_stage2_state(context, user_id)
        if not chapter_id:
            await show_stage2_chapters_menu(chat_id, context)
            return

        if mode == STAGE2_MODE_AWAITING_ANSWER:
            set_stage2_state(
                context,
                chapter_id=chapter_id,
                question_index=stage2_step_index,
                mode=STAGE2_MODE_AWAITING_ANSWER,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            save_stage2_progress(
                user_id,
                chapter_id,
                stage2_step_index,
                STAGE2_MODE_AWAITING_ANSWER,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            await send_stage2_step(
                chat_id,
                context,
                user_id,
                chapter_id,
                stage2_step_index + 1,
                flow_id,
            )
            return

        if mode == STAGE2_MODE_SCENE:
            await send_stage2_step(chat_id, context, user_id, chapter_id, 0, flow_id)
            return

        await send_stage2_finished_menu(chat_id, context, chapter_id, flow_id)
        return

    if current_stage == STAGE_3:
        episode_id, _, mode = get_stage3_state(context, user_id)
        if not episode_id:
            await show_stage3_episodes_menu(chat_id, context)
            return

        if mode == STAGE3_MODE_OPEN_QUESTION:
            set_stage3_state(
                context,
                episode_id=episode_id,
                step=2,
                mode=STAGE3_MODE_FEEDBACK,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            save_stage3_progress(
                user_id,
                episode_id,
                2,
                STAGE3_MODE_FEEDBACK,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            await send_stage3_finished_menu(chat_id, context, user_id, episode_id, flow_id)
            return

        if mode == STAGE3_MODE_TEXT:
            await send_stage3_open_question(chat_id, context, user_id, episode_id, flow_id)
            return

        await send_stage3_finished_menu(chat_id, context, user_id, episode_id, flow_id)
        return

    await skip_stage1_question(
        chat_id,
        context,
        user_id,
        block_id or "intro",
        question_index,
        flow_id,
    )


async def handle_user_response(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    text = sanitize_user_answer(update.message.text or "")

    if text == MENU_BUTTON:
        interrupt_flow(context)
        await show_menu(chat_id, context)
        return

    progress = get_user_progress(user_id)
    current_stage, block_id, question_index, awaiting_answer = get_dialog_state(context, user_id)

    if is_stop_command(text):
        flow_id = interrupt_flow(context)
        await skip_current_step(
            chat_id,
            context,
            user_id,
            current_stage,
            block_id,
            question_index,
            flow_id,
        )
        return

    if current_stage == STAGE_2:
        flow_id = interrupt_flow(context)
        chapter_id, stage2_step_index, mode = get_stage2_state(context, user_id)
        logger.info(
            "[DEBUG] stage2 answer chapter=%s step=%s mode=%s",
            chapter_id,
            stage2_step_index,
            mode,
        )

        if mode != STAGE2_MODE_AWAITING_ANSWER or not chapter_id:
            if mode == STAGE2_MODE_SCENE and chapter_id:
                await send_guarded_message(
                    chat_id,
                    context,
                    flow_id,
                    "Нажми «Продолжить», когда будешь готов перейти к вопросам.",
                    reply_markup=build_stage2_continue_inline_keyboard(),
                )
                return

            if mode == STAGE2_MODE_FINISHED_CHAPTER and chapter_id:
                await send_stage2_finished_menu(chat_id, context, chapter_id, flow_id)
                return

            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Сначала выбери главу этапа 2.",
            )
            start_flow_task(context, show_stage2_chapters_menu(chat_id, context))
            return

        chapter = get_stage2_chapter_map(context).get(chapter_id)
        if not chapter:
            set_stage2_state(context, chapter_id=None, mode=STAGE2_MODE_CHAPTERS)
            save_stage2_progress(user_id, None, 0, STAGE2_MODE_CHAPTERS)
            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Эта глава не найдена. Открою список глав этапа 2.",
            )
            start_flow_task(context, show_stage2_chapters_menu(chat_id, context))
            return

        steps = get_stage2_steps(chapter)
        if stage2_step_index >= len(steps):
            await send_stage2_takeaway(chat_id, context, user_id, chapter_id, flow_id)
            return

        step = steps[stage2_step_index]
        selected_index = parse_stage2_option_answer(text, step["options"])
        if selected_index is None:
            sent = await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Выбери, пожалуйста, один из вариантов: можно написать номер или текст ответа.",
            )
            if not sent:
                return
            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                format_stage2_question(step["prompt"], step["options"]),
            )
            return

        if step["type"] in {"meaning", "line"}:
            is_correct = selected_index == step["correct"]
            fragment_context = get_stage2_fragment_context(chapter, step)
            attempt_count = increase_attempt_count(context, user_id)
            evaluation = evaluate_answer_result(
                user_answer=text,
                answer_type=detect_answer_type(text),
                attempt_count=attempt_count,
                is_correct=is_correct,
                requires_correct=True,
                allowed_points=fragment_context["allowed_points"],
            )
            logger.info(
                "[DEBUG] stage2 advance_allowed=%s attempt_count=%s status=%s",
                evaluation["advance_allowed"],
                attempt_count,
                evaluation["status"],
            )
            recent_replies = trim_recent_replies(context.user_data.get(RECENT_REPLIES_KEY))
            direction = get_stage2_correct_direction(step)
            if evaluation["reached_attempt_limit"] and not is_correct:
                direction = (
                    f"{direction} Если ученик всё ещё не попал в точный смысл, "
                    "коротко объясни главное без давления и помоги спокойно завершить шаг."
                )
            reply_text, should_use_llm, used_llm, guided_answer_type = await generate_guided_reply(
                context_title=str(chapter.get("title") or get_stage2_chapter_label(chapter)),
                context_text=fragment_context["shown_text"],
                question=fragment_context["question"],
                user_answer=text,
                correct_direction=direction,
                learning_goal=fragment_context["learning_goal"],
                allowed_points=fragment_context["allowed_points"],
                forbidden_future_context=fragment_context["forbidden_future_context"],
                answer_options=step.get("options") or [],
                selected_answer=get_stage2_selected_answer(step, selected_index),
                is_correct=is_correct,
                recent_replies=recent_replies,
            )
            logger.info("[DEBUG] stage2_guided_answer_type=%s", guided_answer_type)
            logger.info("[DEBUG] stage2_guided_using_llm=%s", used_llm)
            context.user_data[RECENT_REPLIES_KEY] = trim_recent_replies([*recent_replies, reply_text])

            sent = await send_guarded_message(chat_id, context, flow_id, reply_text)
            if not sent:
                return

            if evaluation["advance_allowed"]:
                set_question_status(context, QUESTION_STATUS_COMPLETED)
                await send_stage2_step(
                    chat_id,
                    context,
                    user_id,
                    chapter_id,
                    stage2_step_index + 1,
                    flow_id,
                )
                return

            set_stage2_state(
                context,
                chapter_id=chapter_id,
                question_index=stage2_step_index,
                mode=STAGE2_MODE_AWAITING_ANSWER,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            save_stage2_progress(
                user_id,
                chapter_id,
                stage2_step_index,
                STAGE2_MODE_AWAITING_ANSWER,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            return

        answer_type = detect_answer_type(text)
        fragment_context = get_stage2_fragment_context(chapter, step)
        attempt_count = increase_attempt_count(context, user_id)
        evaluation = evaluate_answer_result(
            user_answer=get_stage2_selected_answer(step, selected_index) or text,
            answer_type=answer_type,
            attempt_count=attempt_count,
            allowed_points=fragment_context["allowed_points"],
        )
        logger.info(
            "[DEBUG] stage2 personal advance_allowed=%s attempt_count=%s status=%s",
            evaluation["advance_allowed"],
            attempt_count,
            evaluation["status"],
        )
        recent_replies = trim_recent_replies(context.user_data.get(RECENT_REPLIES_KEY))
        personal_direction = "Помочь ученику связать личный выбор с тем, что происходит с героем и как это меняет его понимание."
        if evaluation["reached_attempt_limit"] and not evaluation["is_correct_enough"]:
            personal_direction = (
                f"{personal_direction} Если ответ всё ещё слишком общий, коротко подведи итог "
                "и дай более конкретную опору, но не завершай шаг."
            )
        reply_text, should_use_llm, used_llm, guided_answer_type = await generate_guided_reply(
            context_title=str(chapter.get("title") or get_stage2_chapter_label(chapter)),
            context_text=fragment_context["shown_text"],
            question=fragment_context["question"],
            user_answer=text,
            correct_direction=personal_direction,
            learning_goal=fragment_context["learning_goal"],
            allowed_points=fragment_context["allowed_points"],
            forbidden_future_context=fragment_context["forbidden_future_context"],
            answer_options=step.get("options") or [],
            selected_answer=get_stage2_selected_answer(step, selected_index),
            is_correct=evaluation["advance_allowed"],
            recent_replies=recent_replies,
        )
        logger.info("[DEBUG] stage2_personal_guided_answer_type=%s", guided_answer_type)
        logger.info("[DEBUG] stage2_personal_guided_using_llm=%s", used_llm)
        context.user_data[RECENT_REPLIES_KEY] = trim_recent_replies([*recent_replies, reply_text])

        sent = await send_guarded_message(chat_id, context, flow_id, reply_text)
        if not sent:
            return

        if evaluation["advance_allowed"]:
            set_question_status(context, QUESTION_STATUS_COMPLETED)
            await send_stage2_takeaway(chat_id, context, user_id, chapter_id, flow_id)
            return

        set_stage2_state(
            context,
            chapter_id=chapter_id,
            question_index=stage2_step_index,
            mode=STAGE2_MODE_AWAITING_ANSWER,
            attempt_count=attempt_count,
            question_status=QUESTION_STATUS_AI_FOLLOWUP,
        )
        save_stage2_progress(
            user_id,
            chapter_id,
            stage2_step_index,
            STAGE2_MODE_AWAITING_ANSWER,
            attempt_count=attempt_count,
            question_status=QUESTION_STATUS_AI_FOLLOWUP,
        )
        return

    if current_stage == STAGE_3:
        flow_id = interrupt_flow(context)
        episode_id, stage3_step, mode = get_stage3_state(context, user_id)
        logger.info(
            "[DEBUG] stage3 answer received episode=%s step=%s block=%s",
            episode_id,
            stage3_step,
            mode,
        )

        if mode != STAGE3_MODE_OPEN_QUESTION or not episode_id:
            if mode == STAGE3_MODE_POST_FEEDBACK_NAVIGATION and episode_id:
                await send_stage3_finished_menu(chat_id, context, user_id, episode_id, flow_id)
                return

            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Сначала выбери эпизод этапа 3.",
            )
            start_flow_task(context, show_stage3_episodes_menu(chat_id, context))
            return

        episode = get_stage3_episode_map(context).get(episode_id)
        if not episode:
            set_stage3_state(context, episode_id=None, mode=STAGE3_MODE_EPISODES)
            save_stage3_progress(user_id, None, 0, STAGE3_MODE_EPISODES)
            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Этот эпизод не найден. Открою список эпизодов этапа 3.",
            )
            start_flow_task(context, show_stage3_episodes_menu(chat_id, context))
            return

        answer_type = detect_answer_type(text)
        fragment_context = get_stage3_fragment_context(episode)
        attempt_count = increase_attempt_count(context, user_id)
        evaluation = evaluate_answer_result(
            user_answer=text,
            answer_type=answer_type,
            attempt_count=attempt_count,
            allowed_points=fragment_context["allowed_points"],
        )
        logger.info(
            "[DEBUG] stage3 advance_allowed=%s attempt_count=%s status=%s",
            evaluation["advance_allowed"],
            attempt_count,
            evaluation["status"],
        )
        feedback, used_ai = await generate_stage3_feedback(
            episode_text=fragment_context["shown_text"],
            question=fragment_context["question"],
            user_answer=text,
            core_meanings=fragment_context["allowed_points"],
            good_answer_signals=episode.get("good_answer_signals", []),
            common_mistakes=episode.get("common_mistakes", []),
            reaction_style=episode.get("reaction_style", ""),
        )
        logger.info("[DEBUG] stage3 AI feedback generated=%s", used_ai)
        sent = await send_guarded_message(chat_id, context, flow_id, feedback)
        if not sent:
            return

        if not evaluation["advance_allowed"]:
            set_stage3_state(
                context,
                episode_id=episode_id,
                step=1,
                mode=STAGE3_MODE_OPEN_QUESTION,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            save_stage3_progress(
                user_id,
                episode_id,
                1,
                STAGE3_MODE_OPEN_QUESTION,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            return

        set_stage3_state(
            context,
            episode_id=episode_id,
            step=2,
            mode=STAGE3_MODE_FEEDBACK,
            attempt_count=attempt_count,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        save_stage3_progress(
            user_id,
            episode_id,
            2,
            STAGE3_MODE_FEEDBACK,
            attempt_count=attempt_count,
            question_status=QUESTION_STATUS_COMPLETED,
        )
        await send_stage3_finished_menu(chat_id, context, user_id, episode_id, flow_id)
        return
    if not progress or not progress.get("current_block"):
        flow_id = interrupt_flow(context)
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            "Сначала запусти курс командой /start.",
        )
        return

    block_map = get_block_map(context)
    if current_stage != STAGE_1:
        flow_id = interrupt_flow(context)
        await send_guarded_message(
            chat_id,
            context,
            flow_id,
            STAGE_PLACEHOLDERS.get(
                current_stage,
                "Этот этап скоро будет заполнен. Пока можно вернуться в меню.",
            ),
        )
        return

    block_id = block_id or progress["current_block"]
    if block_id not in block_map:
        block_id = "intro"
        question_index = 0
        awaiting_answer = False

    block = block_map[block_id]
    question_index = max(0, min(question_index, len(block["questions"]) - 1))
    current_question = block["questions"][question_index]

    logger.info("[DEBUG] block_id=%s", block_id)
    logger.info("[DEBUG] question_index=%s", question_index)
    logger.info('[DEBUG] user_answer="%s"', text)
    logger.info("[DEBUG] awaiting_contents_choice=%s", is_contents_mode(context))
    logger.info("[DEBUG] awaiting_answer=%s", awaiting_answer)

    flow_id = interrupt_flow(context)
    answer_type = detect_answer_type(text)
    logger.info("[DEBUG] answer_type=%s", answer_type)

    async def process_answer_flow() -> None:
        fragment_context = get_stage1_fragment_context(block, current_question)
        attempt_count = increase_attempt_count(context, user_id)
        if answer_type == "dont_know":
            increase_dont_know_count(context)
        else:
            reset_dont_know_count(context)

        evaluation = evaluate_answer_result(
            user_answer=text,
            answer_type=answer_type,
            attempt_count=attempt_count,
            allowed_points=fragment_context["allowed_points"],
        )
        logger.info(
            "[DEBUG] stage1 advance_allowed=%s attempt_count=%s status=%s",
            evaluation["advance_allowed"],
            attempt_count,
            evaluation["status"],
        )
        recent_replies = trim_recent_replies(context.user_data.get(RECENT_REPLIES_KEY))
        direction = fragment_context["learning_goal"] or "Помочь увидеть, что происходит с Петром в уже показанном фрагменте."
        if evaluation["reached_attempt_limit"] and not evaluation["is_correct_enough"]:
            direction = (
                f"{direction} Если ученик всё ещё затрудняется, коротко подведи итог "
                "и дай более конкретную опору, чтобы спокойно завершить шаг."
            )
        reply_text, should_use_llm, used_llm, _ = await generate_guided_reply(
            context_title=get_block_title(block),
            context_text=fragment_context["shown_text"],
            question=fragment_context["question"],
            user_answer=text,
            correct_direction=direction,
            learning_goal=fragment_context["learning_goal"],
            allowed_points=fragment_context["allowed_points"],
            forbidden_future_context=fragment_context["forbidden_future_context"],
            is_correct=evaluation["advance_allowed"],
            recent_replies=recent_replies,
        )

        logger.info("[DEBUG] should_use_llm=%s", should_use_llm)
        logger.info("[DEBUG] using_llm=%s", used_llm)

        context.user_data[RECENT_REPLIES_KEY] = trim_recent_replies(
            [*recent_replies, reply_text]
        )

        sent = await send_guarded_message(
            chat_id,
            context,
            flow_id,
            reply_text,
        )
        if not sent:
            return

        if not evaluation["advance_allowed"]:
            set_dialog_state(
                context,
                current_stage=STAGE_1,
                current_block=block_id,
                question_index=question_index,
                awaiting_answer=True,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            save_user_progress(
                user_id,
                block_id,
                question_index,
                STAGE_1,
                attempt_count=attempt_count,
                question_status=QUESTION_STATUS_AI_FOLLOWUP,
            )
            return

        set_question_status(context, QUESTION_STATUS_COMPLETED)
        next_question_index = question_index + 1
        if next_question_index < len(block["questions"]):
            save_user_progress(
                user_id,
                block_id,
                next_question_index,
                STAGE_1,
                attempt_count=0,
                question_status=QUESTION_STATUS_WAITING,
            )
            set_dialog_state(
                context,
                current_stage=STAGE_1,
                current_block=block_id,
                question_index=next_question_index,
                awaiting_answer=True,
                attempt_count=0,
                question_status=QUESTION_STATUS_WAITING,
            )
            await send_current_question(
                chat_id,
                context,
                block_id,
                next_question_index,
                flow_id,
            )
            return

        next_block_id = block["next"]
        if not next_block_id:
            save_user_progress(
                user_id,
                None,
                0,
                STAGE_1,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            set_dialog_state(
                context,
                current_stage=STAGE_1,
                current_block=None,
                question_index=0,
                awaiting_answer=False,
                attempt_count=0,
                question_status=QUESTION_STATUS_COMPLETED,
            )
            await send_guarded_message(
                chat_id,
                context,
                flow_id,
                "Это был последний блок. Если хочешь пройти всё заново, открой меню.",
            )
            return

        save_user_progress(
            user_id,
            next_block_id,
            0,
            STAGE_1,
            attempt_count=0,
            question_status=QUESTION_STATUS_WAITING,
        )
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block=next_block_id,
            question_index=0,
            awaiting_answer=False,
            attempt_count=0,
            question_status=QUESTION_STATUS_WAITING,
        )
        await send_block(chat_id, context, next_block_id, flow_id)

    start_flow_task(context, process_answer_flow())


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data

    logger.info("[DEBUG] callback received: %s", data)

    if data == CALLBACK_MENU_CONTINUE:
        logger.info("[DEBUG] continue pressed")
        user_id = update.effective_user.id
        current_stage, current_block, question_index, awaiting_answer = get_dialog_state(
            context, user_id
        )
        logger.info(
            "[DEBUG] restoring stage=%s block=%s question=%s awaiting=%s",
            current_stage,
            current_block,
            question_index,
            awaiting_answer,
        )

        if current_stage == STAGE_2:
            flow_id = interrupt_flow(context)
            chapter_id, stage2_step_index, mode = get_stage2_state(context, user_id)
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Продолжаем 👇",
            )
            if mode in {STAGE2_MODE_SCENE, STAGE2_MODE_AWAITING_ANSWER} and chapter_id:
                reset_attempts = mode == STAGE2_MODE_SCENE
                start_flow_task(
                    context,
                    send_stage2_step(
                        query.message.chat_id,
                        context,
                        user_id,
                        chapter_id,
                        stage2_step_index,
                        flow_id,
                        reset_attempts=reset_attempts,
                    ),
                )
                return
            if mode == STAGE2_MODE_FINISHED_CHAPTER and chapter_id:
                start_flow_task(
                    context,
                    send_stage2_finished_menu(
                        query.message.chat_id,
                        context,
                        chapter_id,
                        flow_id,
                    ),
                )
                return
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        if current_stage == STAGE_3:
            flow_id = interrupt_flow(context)
            episode_id, stage3_step, mode = get_stage3_state(context, user_id)
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Продолжаем 👇",
            )
            if mode == STAGE3_MODE_AWAITING_ANSWER and episode_id:
                start_flow_task(
                    context,
                    send_stage3_open_question(
                        query.message.chat_id,
                        context,
                        user_id,
                        episode_id,
                        flow_id,
                        reset_attempts=False,
                    ),
                )
                return
            if mode == STAGE3_MODE_FINISHED_EPISODE and episode_id:
                start_flow_task(
                    context,
                    send_stage3_finished_menu(
                        query.message.chat_id,
                        context,
                        user_id,
                        episode_id,
                        flow_id,
                    ),
                )
                return
            start_flow_task(context, show_stage3_episodes_menu(query.message.chat_id, context))
            return

        if current_stage != STAGE_1:
            flow_id = interrupt_flow(context)
            set_current_stage(context, current_stage)
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                STAGE_PLACEHOLDERS.get(
                    current_stage,
                    "Этот этап скоро будет заполнен. Пока можно вернуться в меню.",
                ),
            )
            return

        if not current_block:
            flow_id = interrupt_flow(context)
            save_user_progress(user_id, "intro", 0, STAGE_1)
            set_dialog_state(
                context,
                current_stage=STAGE_1,
                current_block="intro",
                question_index=0,
                awaiting_answer=False,
            )
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Продолжаем 👇",
            )
            start_flow_task(context, send_block(query.message.chat_id, context, "intro", flow_id))
            return

        block_map = get_block_map(context)
        if current_block not in block_map:
            current_block = "intro"
            question_index = 0
            awaiting_answer = False

        block = block_map[current_block]
        if question_index >= len(block["questions"]):
            next_block_id = block.get("next") or "intro"
            current_block = next_block_id
            question_index = 0
            awaiting_answer = False

        flow_id = interrupt_flow(context)
        set_contents_mode(context, False)
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block=current_block,
            question_index=question_index,
            awaiting_answer=awaiting_answer,
        )
        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            "Продолжаем 👇",
        )

        if awaiting_answer:
            start_flow_task(
                context,
                send_current_question(
                    query.message.chat_id,
                    context,
                    current_block,
                    question_index,
                    flow_id,
                    reset_attempts=False,
                    user_id=user_id,
                ),
            )
        else:
            start_flow_task(
                context,
                send_block(query.message.chat_id, context, current_block, flow_id),
            )
        return

    if data == CALLBACK_MENU_HELP:
        flow_id = interrupt_flow(context)
        logger.info("[DEBUG] callback received: help")
        help_text = 'ℹ️ Как пользоваться ботом\n\n1. Выбери этап:\n\n   1. Герои и сюжет — чтобы понять, кто есть кто\n   2. Пересказ — чтобы разобраться в событиях\n   3. С цитатами — чтобы углубить понимание\n\n2. Читай последовательно и отвечай на вопросы.\n   Бот поможет, если ответ неточный.\n\n3. Используй:\n   «Продолжить» — чтобы вернуться к последнему месту\n   «Меню» — чтобы выбрать другой раздел\n   «стоп» — чтобы пропустить вопрос и перейти дальше\n\n4. Можно проходить в своём темпе.\n   Лучше идти по порядку: 1 → 2 → 3'
        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            help_text,
            reply_markup=build_menu_only_inline_keyboard(),
        )
        return

    if data == CALLBACK_MENU_ORIGINAL:
        flow_id = interrupt_flow(context)
        logger.info("[DEBUG] callback received: original")
        original_text = (
            "Ты уже здорово разобрался в истории и героях.\n"
            "Теперь читать оригинал будет намного легче и интереснее.\n\n"
            "Попробуй прочитать книгу целиком:\n\n"
            "https://ilibrary.ru/text/107/p.1/index.html"
        )
        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            original_text,
            reply_markup=build_menu_only_inline_keyboard(),
        )
        return

    if data == CALLBACK_MENU_OPEN:
        flow_id = interrupt_flow(context)
        logger.info("[DEBUG] callback received: menu_open")
        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            "Открываю меню 👇",
        )
        start_flow_task(context, show_menu(query.message.chat_id, context))
        return

    if data.startswith(STAGE_CALLBACK_PREFIX):
        stage_id = data[len(STAGE_CALLBACK_PREFIX) :]
        logger.info("[DEBUG] stage selected: %s", stage_id)
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        set_contents_mode(context, False)
        reset_recent_replies(context)
        reset_dont_know_count(context)

        if stage_id == STAGE_1:
            save_user_progress(user_id, None, 0, STAGE_1)
            set_dialog_state(
                context,
                current_stage=STAGE_1,
                current_block=None,
                question_index=0,
                awaiting_answer=False,
            )
            logger.info("[DEBUG] flow interrupted by menu action")
            start_flow_task(context, show_stage_1_contents_menu(query.message.chat_id, context))
            return

        if stage_id == STAGE_2:
            save_stage2_progress(user_id, None, 0, STAGE2_MODE_CHAPTERS)
            set_stage2_state(
                context,
                chapter_id=None,
                question_index=0,
                mode=STAGE2_MODE_CHAPTERS,
            )
            logger.info("[DEBUG] flow interrupted by menu action")
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        if stage_id == STAGE_3:
            save_stage3_progress(user_id, None, 0, STAGE3_MODE_EPISODES)
            set_stage3_state(
                context,
                episode_id=None,
                step=0,
                mode=STAGE3_MODE_EPISODES,
            )
            logger.info("[DEBUG] flow interrupted by menu action")
            start_flow_task(context, show_stage3_episodes_menu(query.message.chat_id, context))
            return


        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            "Такого этапа нет. Открой меню и выбери пункт из списка.",
        )
        return

    if data.startswith(BLOCK_CALLBACK_PREFIX):
        block_id = data[len(BLOCK_CALLBACK_PREFIX) :]
        logger.info("[DEBUG] block jump: %s", block_id)
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        block_map = get_block_map(context)

        if block_id not in block_map:
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Такой главы нет. Открой меню и выбери главу из списка.",
            )
            return

        save_user_progress(user_id, block_id, 0, STAGE_1)
        set_contents_mode(context, False)
        reset_recent_replies(context)
        reset_dont_know_count(context)
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block=block_id,
            question_index=0,
            awaiting_answer=False,
        )
        await send_guarded_message(
            query.message.chat_id,
            context,
            flow_id,
            "Переходим к выбранной главе 👇",
        )
        start_flow_task(context, send_block(query.message.chat_id, context, block_id, flow_id))
        return

    if data.startswith(STAGE2_CHAPTER_PREFIX):
        chapter_id = data[len(STAGE2_CHAPTER_PREFIX) :]
        logger.info("[DEBUG] stage2 chapter selected: %s", chapter_id)
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        if chapter_id not in get_stage2_chapter_map(context):
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Такой главы этапа 2 нет. Открою список глав.",
            )
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        start_flow_task(
            context,
            send_stage2_chapter(query.message.chat_id, context, user_id, chapter_id, flow_id),
        )
        return

    if data == CALLBACK_STAGE2_NEXT:
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        chapter_id, _, _ = get_stage2_state(context, user_id)
        next_chapter_id = get_next_stage2_chapter_id(context, chapter_id or "")
        if not next_chapter_id:
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Следующей главы нет. Открою список глав этапа 2.",
            )
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        start_flow_task(
            context,
            send_stage2_chapter(
                query.message.chat_id,
                context,
                user_id,
                next_chapter_id,
                flow_id,
            ),
        )
        return

    if data == CALLBACK_STAGE2_CONTINUE:
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        chapter_id, stage2_step_index, mode = get_stage2_state(context, user_id)
        if not chapter_id:
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        if mode == STAGE2_MODE_FINISHED_CHAPTER:
            next_chapter_id = get_next_stage2_chapter_id(context, chapter_id)
            if next_chapter_id:
                start_flow_task(
                    context,
                    send_stage2_chapter(
                        query.message.chat_id,
                        context,
                        user_id,
                        next_chapter_id,
                        flow_id,
                    ),
                )
                return
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "2. Пересказ завершён. Можно вернуться к списку эпизодов.",
            )
            start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
            return

        start_flow_task(
            context,
            send_stage2_step(
                query.message.chat_id,
                context,
                user_id,
                chapter_id,
                stage2_step_index,
                flow_id,
            ),
        )
        return

    if data == CALLBACK_STAGE2_CHAPTERS:
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        save_stage2_progress(user_id, None, 0, STAGE2_MODE_CHAPTERS)
        set_stage2_state(context, chapter_id=None, question_index=0, mode=STAGE2_MODE_CHAPTERS)
        start_flow_task(context, show_stage2_chapters_menu(query.message.chat_id, context))
        return


    if data.startswith(STAGE3_EPISODE_PREFIX):
        episode_id = data[len(STAGE3_EPISODE_PREFIX) :]
        logger.info("[DEBUG] stage3 episode selected: %s", episode_id)
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        if episode_id not in get_stage3_episode_map(context):
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "Такого эпизода этапа 3 нет. Открою список эпизодов.",
            )
            start_flow_task(context, show_stage3_episodes_menu(query.message.chat_id, context))
            return

        start_flow_task(
            context,
            send_stage3_episode(query.message.chat_id, context, user_id, episode_id, flow_id),
        )
        return

    if data == CALLBACK_STAGE3_NEXT:
        logger.info("[DEBUG] stage3 navigation selected: continue")
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        episode_id, _, _ = get_stage3_state(context, user_id)
        next_episode_id = get_next_stage3_episode_id(context, episode_id or "")
        if not next_episode_id:
            await send_guarded_message(
                query.message.chat_id,
                context,
                flow_id,
                "3. С цитатами уже завершён. Открою список эпизодов.",
            )
            start_flow_task(context, show_stage3_episodes_menu(query.message.chat_id, context))
            return

        start_flow_task(
            context,
            send_stage3_episode(
                query.message.chat_id,
                context,
                user_id,
                next_episode_id,
                flow_id,
            ),
        )
        return

    if data == CALLBACK_STAGE3_EPISODES:
        logger.info("[DEBUG] stage3 navigation selected: episodes")
        flow_id = interrupt_flow(context)
        user_id = update.effective_user.id
        save_stage3_progress(user_id, None, 0, STAGE3_MODE_EPISODES)
        set_stage3_state(context, episode_id=None, step=0, mode=STAGE3_MODE_EPISODES)
        start_flow_task(context, show_stage3_episodes_menu(query.message.chat_id, context))
        return
