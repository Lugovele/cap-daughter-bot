import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROGRESS_FILE = DATA_DIR / "progress.json"


def _ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text("{}", encoding="utf-8")


def _read_progress() -> dict:
    _ensure_storage()
    with PROGRESS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_progress(data: dict) -> None:
    _ensure_storage()
    with PROGRESS_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_user_progress(user_id: int) -> dict | None:
    data = _read_progress()
    progress = data.get(str(user_id))
    if progress and "current_stage" not in progress:
        progress["current_stage"] = "stage_1"
    return progress


def save_user_progress(
    user_id: int,
    current_block: str | None,
    question_index: int = 0,
    current_stage: str = "stage_1",
    **extra_fields,
) -> None:
    data = _read_progress()
    data[str(user_id)] = {
        "current_stage": current_stage,
        "current_block": current_block,
        "question_index": question_index,
        **extra_fields,
    }
    _write_progress(data)


def reset_user_progress(user_id: int) -> None:
    save_user_progress(
        user_id=user_id,
        current_block="intro",
        question_index=0,
        current_stage="stage_1",
    )


def save_stage2_progress(
    user_id: int,
    current_chapter_id: str | None,
    current_question_index: int = 0,
    current_mode: str = "chapters",
    attempt_count: int = 0,
    question_status: str = "waiting_user_answer",
    current_question: str | None = None,
    question_type: str = "choice",
    current_block_id: str | None = None,
) -> None:
    save_user_progress(
        user_id=user_id,
        current_block=current_block_id or current_chapter_id,
        question_index=current_question_index,
        current_stage="stage_2",
        current_work="captains_daughter",
        current_episode=current_chapter_id,
        current_step=current_question_index,
        current_stage2_block=current_mode,
        current_question_type=question_type,
        current_attempt_state=question_status,
        current_question=current_question,
        current_question_id=current_question_index,
        current_block_id=current_block_id or current_chapter_id,
        current_chapter_id=current_chapter_id,
        current_question_index=current_question_index,
        current_mode=current_mode,
        attempt_count=attempt_count,
        question_status=question_status,
    )


def save_stage3_progress(
    user_id: int,
    current_episode_id: str | None,
    current_step: int = 0,
    current_mode: str = "episodes",
    attempt_count: int = 0,
    question_status: str = "waiting_user_answer",
) -> None:
    save_user_progress(
        user_id=user_id,
        current_block=current_mode,
        question_index=0,
        current_stage="stage_3",
        current_work="captains_daughter",
        current_episode=current_episode_id,
        current_step=current_step,
        current_stage3_block=current_mode,
        current_question_type=current_mode,
        current_attempt_state=question_status,
        current_episode_id=current_episode_id,
        current_mode=current_mode,
        attempt_count=attempt_count,
        question_status=question_status,
    )
