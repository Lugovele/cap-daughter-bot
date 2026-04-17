from telegram import Update
from telegram.ext import ContextTypes

from handlers.story import (
    STAGE_1,
    STAGE_TITLES,
    build_main_keyboard,
    get_current_stage,
    get_block_map,
    get_block_title,
    interrupt_flow,
    reset_dont_know_count,
    reset_recent_replies,
    send_block,
    send_current_question,
    set_dialog_state,
    set_contents_mode,
    start_flow_task,
)
from services.message_sender import send_typing_message
from services.progress import get_user_progress, reset_user_progress


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    saved_progress = get_user_progress(user_id)
    saved_stage = (saved_progress or {}).get("current_stage", STAGE_1)
    current_block = (saved_progress or {}).get("current_block") or "intro"
    if saved_stage != STAGE_1:
        reset_user_progress(user_id)
        saved_progress = get_user_progress(user_id)
        current_block = "intro"
    flow_id = interrupt_flow(context)
    set_contents_mode(context, False)

    if saved_progress is None:
        reset_user_progress(user_id)
        reset_recent_replies(context)
        reset_dont_know_count(context)
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block="intro",
            question_index=0,
            awaiting_answer=False,
        )
        await send_typing_message(
            update.effective_chat.id,
            "Привет. Мы будем изучать «Капитанскую дочку» по шагам.",
            reply_markup=build_main_keyboard(),
        )
        start_flow_task(
            context,
            send_block(update.effective_chat.id, context, "intro", flow_id),
        )
        return

    if not current_block:
        reset_user_progress(user_id)
        reset_recent_replies(context)
        reset_dont_know_count(context)
        set_dialog_state(
            context,
            current_stage=STAGE_1,
            current_block="intro",
            question_index=0,
            awaiting_answer=False,
        )
        await send_typing_message(
            update.effective_chat.id,
            "Начнём сначала.",
            reply_markup=build_main_keyboard(),
        )
        start_flow_task(
            context,
            send_block(update.effective_chat.id, context, "intro", flow_id),
        )
        return

    block = get_block_map(context)[current_block]
    current_question = saved_progress.get("question_index", 0)
    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block=current_block,
        question_index=current_question,
        awaiting_answer=True,
    )

    await send_typing_message(
        update.effective_chat.id,
        f"Продолжим раздел: {get_block_title(block)}.",
        reply_markup=build_main_keyboard(),
    )
    start_flow_task(
        context,
        send_current_question(
            update.effective_chat.id,
            context,
            current_block,
            current_question,
            flow_id,
        ),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "/start — начать или продолжить курс\n"
        "/help — показать помощь\n"
        "/progress — показать текущий блок\n"
        "/restart — начать заново"
    )
    await send_typing_message(
        update.effective_chat.id,
        help_text,
        reply_markup=build_main_keyboard(),
    )


async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    progress = get_user_progress(user_id)

    if not progress:
        await send_typing_message(
            update.effective_chat.id,
            "Прогресс пока пуст. Используй /start, чтобы начать.",
            reply_markup=build_main_keyboard(),
        )
        return

    current_stage = get_current_stage(context, user_id)
    if current_stage != STAGE_1:
        await send_typing_message(
            update.effective_chat.id,
            f"Сейчас твой этап: {STAGE_TITLES.get(current_stage, current_stage)}.",
            reply_markup=build_main_keyboard(),
        )
        return

    if not progress.get("current_block"):
        await send_typing_message(
            update.effective_chat.id,
            "Сейчас выбран этап 1. Открой меню и выбери главу в содержании.",
            reply_markup=build_main_keyboard(),
        )
        return

    await send_typing_message(
        update.effective_chat.id,
        (
            f"Сейчас твой этап: "
            f"{STAGE_TITLES[STAGE_1]}. "
            f"Раздел: "
            f"{get_block_title(get_block_map(context)[progress['current_block']])}. "
            f"Текущий вопрос: {progress['question_index'] + 1}"
        ),
        reply_markup=build_main_keyboard(),
    )


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    reset_user_progress(user_id)
    set_contents_mode(context, False)
    reset_recent_replies(context)
    reset_dont_know_count(context)
    flow_id = interrupt_flow(context)
    set_dialog_state(
        context,
        current_stage=STAGE_1,
        current_block="intro",
        question_index=0,
        awaiting_answer=False,
    )
    await send_typing_message(
        update.effective_chat.id,
        "Начинаем заново. Возвращаемся к первой части истории.",
        reply_markup=build_main_keyboard(),
    )
    start_flow_task(
        context,
        send_block(update.effective_chat.id, context, "intro", flow_id),
    )
