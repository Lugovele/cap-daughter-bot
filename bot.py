import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from handlers.start import help_command, progress_command, restart_command, start_command
from handlers.story import handle_callback, handle_user_response
from services.message_sender import configure_bot


BASE_DIR = Path(__file__).resolve().parent
CONTENT_PATH = BASE_DIR / "content" / "captains_daughter.json"
STAGE2_CONTENT_PATH = BASE_DIR / "content" / "stage2_retelling.json"


logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def load_course() -> dict:
    with CONTENT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_stage2_content() -> dict:
    with STAGE2_CONTENT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


async def post_init(application: Application) -> None:
    configure_bot(application.bot)
    application.bot_data["course"] = load_course()
    application.bot_data["stage_2"] = load_stage2_content()
    logger.info("Course loaded: %s", application.bot_data["course"]["course_title"])
    logger.info("Stage 2 loaded: %s", application.bot_data["stage_2"]["title"])


def build_application() -> Application:
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set. Add it to the .env file.")

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("progress", progress_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_response)
    )

    return application


def main() -> None:
    application = build_application()
    logger.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
