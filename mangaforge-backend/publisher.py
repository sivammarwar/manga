"""
publisher.py
============
Daily chapter publisher for MangaForge AI.
Runs every day at 8 AM IST via GitHub Actions.

Finds chapters scheduled for today, validates they have real content + images,
and marks them as published so the Next.js frontend can display them.
"""

import os
import json
import logging
from datetime import date, datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("publisher.log")],
)
log = logging.getLogger("publisher")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def get_prose(content: str) -> str:
    """Extract prose text from chapter content (JSON or plain text)."""
    if not content:
        return ""
    try:
        data = json.loads(content)
        return data.get("prose", "")
    except Exception:
        return content


def get_todays_chapters() -> list[dict]:
    today = date.today().isoformat()
    try:
        result = (
            supabase.table("chapters")
            .select("*, articles(title, category, status), images(id, image_url)")
            .eq("scheduled_date", today)
            .neq("status", "published")
            .execute()
        )
        return result.data or []
    except Exception as e:
        log.error("❌ Failed to fetch today's chapters: %s", e)
        return []


def validate_chapter(chapter: dict) -> tuple[bool, str]:
    prose = get_prose(chapter.get("content", ""))
    if len(prose) < 100:
        return False, "no real content yet"

    images = chapter.get("images", [])
    real_images = [i for i in images if "picsum" not in i.get("image_url", "")]
    if not real_images:
        return False, "images still generating"

    return True, "ok"


def publish_chapter(chapter: dict) -> bool:
    try:
        supabase.table("chapters").update({
            "status"      : "published",
            "published"   : True,
            "published_at": datetime.utcnow().isoformat(),
        }).eq("id", chapter["id"]).execute()
        return True
    except Exception as e:
        log.error("  ❌ Failed to publish chapter %s: %s", chapter["id"], e)
        return False


def run():
    today = date.today().isoformat()
    log.info("")
    log.info("=" * 60)
    log.info("📢  MANGAFORGE DAILY PUBLISHER  |  %s", today)
    log.info("=" * 60)

    chapters = get_todays_chapters()

    if not chapters:
        log.info("ℹ️  No chapters scheduled for today.")
        return

    log.info("Found %d chapters scheduled for today.\n", len(chapters))

    published_list = []
    skipped_list   = []

    for ch in chapters:
        art      = ch.get("articles") or {}
        title    = art.get("title", "Unknown")[:40]
        category = art.get("category", "?")
        ch_num   = ch.get("chapter_number", "?")

        valid, reason = validate_chapter(ch)

        if not valid:
            log.warning("  ⏭️  Skipping  [%s] Ch.%s — %s", title, ch_num, reason)
            skipped_list.append(ch)
            continue

        if publish_chapter(ch):
            log.info("  ✅ Published  [%s · %s] Chapter %s", category.upper(), title, ch_num)
            published_list.append(ch)
        else:
            skipped_list.append(ch)

    log.info("")
    log.info("─" * 60)
    log.info("📊  Published : %d   Skipped : %d", len(published_list), len(skipped_list))

    if skipped_list:
        log.info("    Skipped (will retry tomorrow):")
        for ch in skipped_list:
            art = ch.get("articles") or {}
            log.info("      • %s  Ch.%s", art.get("title", "?")[:35], ch.get("chapter_number"))

    log.info("=" * 60)


if __name__ == "__main__":
    run()