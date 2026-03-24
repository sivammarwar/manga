"""
pipeline.py
===========
MangaForge AI — Full Pipeline (runs every Monday 6AM IST via GitHub Actions)

Flow:
  Phase 1 — Ideation: Use uploaded manga topics from DB (or fallback) to create series + characters
  Phase 2 — Chapter Outlines: Pre-plan ALL 21 chapter summaries with start/end hooks + image prompts
  Phase 3 — Write chapters: Full human-like content using outlines, ends each with suspense
  Phase 4 — Generate images: HF API with 4-token rotation, precise prompts per scene
  Phase 5 — Publish: Mark chapters as published once content + images are ready

Everything runs on Day 1. Daily publisher just re-publishes any that were missed.
"""

import os
import json
import time
import re
import hashlib
import logging
import requests
import boto3
from botocore.config import Config
from datetime import datetime, timedelta, date

from google import genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
log = logging.getLogger("pipeline")

# ─── Clients ──────────────────────────────────────────────────────────────────

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ─── Gemini Models (fallback chain) ───────────────────────────────────────────

GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

# ─── HuggingFace — 4 token rotation ──────────────────────────────────────────

HF_TOKENS = [
    os.getenv("HF_API_TOKEN_1") or os.getenv("HF_API_TOKEN"),
    os.getenv("HF_API_TOKEN_2"),
    os.getenv("HF_API_TOKEN_3"),
    os.getenv("HF_API_TOKEN_4"),
]
HF_TOKENS = [t for t in HF_TOKENS if t]  # remove None

HF_MODELS = [
    "dreamlike-art/dreamlike-anime-1-0",
    "Linaqruf/anything-v3-better-vae",
    "hakurei/waifu-diffusion",
]

STYLE_POS = (
    "anime style, manga panel, detailed background, "
    "vibrant colors, masterpiece, best quality, sharp focus, "
    "cinematic lighting, expressive faces"
)
STYLE_NEG = (
    "lowres, bad anatomy, worst quality, blurry, deformed, "
    "ugly, text, watermark, western style, 3d render, extra limbs"
)

# ─── R2 Config ────────────────────────────────────────────────────────────────

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS   = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET   = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET   = os.getenv("R2_BUCKET_NAME", "manga-images")
R2_PUBLIC   = os.getenv("R2_PUBLIC_URL")

CATEGORIES = [
    "romantic", "action", "emotional", "sad", "spicy",
    "horror", "comedy", "fantasy", "mystery", "scifi"
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def gemini_generate(prompt: str, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        for model in GEMINI_MODELS:
            try:
                resp = gemini.models.generate_content(model=model, contents=prompt)
                return resp.text
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    time.sleep(3)
                    continue
                if "429" in err or "QUOTA" in err.upper():
                    log.warning("Gemini quota hit on %s — trying next model", model)
                    time.sleep(5)
                    continue
                log.error("Gemini error (%s): %s", model, err[:80])
        if attempt < max_retries - 1:
            time.sleep(8)
    return None


def clean_json(text: str) -> dict | list | None:
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # find outermost { } or [ ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end   = text.rfind(end_char) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None


def safe_int(value, default=20):
    try:
        return int(float(str(value).replace(".0", "")))
    except Exception:
        return default


# ─── HF image generation with 4-token rotation ───────────────────────────────

_hf_token_idx = 0
_hf_model_idx = 0


def hf_generate_image(prompt: str) -> bytes | None:
    global _hf_token_idx, _hf_model_idx

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": STYLE_NEG,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
        }
    }

    # Try every combination of token × model (4 × 3 = 12 attempts max)
    for _ in range(len(HF_TOKENS) * len(HF_MODELS)):
        token = HF_TOKENS[_hf_token_idx % len(HF_TOKENS)]
        model = HF_MODELS[_hf_model_idx % len(HF_MODELS)]
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 20)
                log.info("  ⏳ HF model loading — waiting %ds", wait)
                time.sleep(wait + 5)
                _hf_model_idx += 1
                continue
            if resp.status_code == 429:
                log.warning("  ⚠️  Token %d rate-limited — rotating", _hf_token_idx % len(HF_TOKENS))
                _hf_token_idx += 1
                time.sleep(3)
                continue
            log.error("  HF HTTP %d: %s", resp.status_code, resp.text[:80])
            _hf_model_idx += 1
        except Exception as e:
            log.error("  HF request error: %s", e)
            time.sleep(5)
            _hf_token_idx += 1

    log.error("  ❌ All HF tokens + models exhausted for this image")
    return None


def upload_to_r2(image_bytes: bytes, filename: str) -> str | None:
    try:
        r2 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS,
            aws_secret_access_key=R2_SECRET,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        key = f"chapters/{filename}"
        r2.put_object(
            Bucket=R2_BUCKET, Key=key, Body=image_bytes,
            ContentType="image/png", CacheControl="public, max-age=31536000"
        )
        return f"{R2_PUBLIC}/{key}"
    except Exception as e:
        log.error("R2 upload error: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MangaForgePipeline:

    def __init__(self):
        self.week_start = date.today()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — IDEATION (use uploaded topics from DB or Gemini fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def run_phase_1_ideation(self):
        log.info("=" * 60)
        log.info("🚀  Phase 1: Ideation  |  Week: %s", self.week_start)
        log.info("=" * 60)

        # Skip if already ran this week
        existing = (
            supabase.table("pipeline_runs")
            .select("id")
            .eq("week_start", self.week_start.isoformat())
            .execute()
        )
        if existing.data:
            log.info("Pipeline already exists for this week — skipping Phase 1.")
            return

        supabase.table("pipeline_runs").insert({
            "week_start"   : self.week_start.isoformat(),
            "phase"        : "ideation",
            "phase_progress": 0,
            "series_count" : 10,
        }).execute()

        # Fetch uploaded topics from DB (uploaded via admin panel)
        uploaded_topics = (
            supabase.table("manga_topics")
            .select("*")
            .eq("used", False)
            .limit(10)
            .execute()
            .data
        ) or []

        topics_by_category = {t["category"]: t for t in uploaded_topics}

        for i, category in enumerate(CATEGORIES):
            log.info("[%d/10] Generating %s series...", i + 1, category)

            topic_row = topics_by_category.get(category)
            topic_hint = ""
            if topic_row:
                topic_hint = f"\n\nIMPORTANT — Use this specific story summary provided by the creator:\n{topic_row['summary']}"
                # Mark topic as used
                supabase.table("manga_topics").update({"used": True}).eq("id", topic_row["id"]).execute()

            premise_prompt = f"""You are a professional manga writer. Create a compelling manga series premise for the "{category}" genre.{topic_hint}

Return ONLY valid JSON (no markdown, no extra text):
{{
    "title": "An evocative, original series title",
    "logline": "A single gripping sentence that hooks the reader",
    "global_prompt": "2-3 sentence series overview: setting, core conflict, emotional stakes",
    "tone": "e.g. dark and brooding, lighthearted and warm, etc.",
    "characters": [
        {{
            "name": "Character full name",
            "age": 22,
            "gender": "female",
            "description": "Physical appearance in detail — hair, eyes, clothing style",
            "personality": ["trait1", "trait2", "trait3"],
            "role": "protagonist/antagonist/side character",
            "backstory": "1-2 sentence backstory that drives their motivation"
        }},
        {{
            "name": "Second character name",
            "age": 24,
            "gender": "male",
            "description": "Physical appearance",
            "personality": ["trait1", "trait2", "trait3"],
            "role": "protagonist/love interest/rival",
            "backstory": "1-2 sentence backstory"
        }}
    ]
}}"""

            try:
                raw = gemini_generate(premise_prompt)
                if not raw:
                    log.warning("No response for %s — skipping", category)
                    continue

                data = clean_json(raw)
                if not data:
                    log.warning("Bad JSON for %s — skipping", category)
                    continue

                # Insert article
                article_res = supabase.table("articles").insert({
                    "title"        : str(data.get("title", f"{category.title()} Series"))[:150],
                    "category"     : category,
                    "logline"      : str(data.get("logline", ""))[:300],
                    "global_prompt": str(data.get("global_prompt", ""))[:1000],
                    "status"       : "character_gen",
                    "week_start"   : self.week_start.isoformat(),
                }).execute()

                article_id = article_res.data[0]["id"]

                # Insert characters with rich prompts for image consistency
                for char in data.get("characters", [])[:2]:
                    age = safe_int(char.get("age"), 20)
                    name = str(char.get("name", "Unknown"))[:100]
                    desc = str(char.get("description", ""))[:300]
                    personality = [str(p)[:50] for p in char.get("personality", [])[:3]]
                    gender = str(char.get("gender", "other"))[:20]

                    # Build a master_prompt that will be prepended to every image
                    master_prompt = (
                        f"anime character, {name}, {gender}, age {age}, "
                        f"{desc}, {', '.join(personality)}, "
                        "masterpiece, best quality, detailed face, consistent character design"
                    )

                    supabase.table("characters").insert({
                        "article_id"  : article_id,
                        "name"        : name,
                        "age"         : age,
                        "gender"      : gender,
                        "description" : desc,
                        "personality" : personality,
                        "master_prompt": master_prompt[:500],
                    }).execute()

                progress = ((i + 1) / 10) * 100
                supabase.table("pipeline_runs").update({
                    "phase_progress": progress
                }).eq("week_start", self.week_start.isoformat()).execute()

                log.info("  ✓ %s: %s", category, data.get("title"))

            except Exception as e:
                log.error("  Error on %s: %s", category, e)

            time.sleep(2)

        supabase.table("pipeline_runs").update({
            "phase": "chapter_outlines"
        }).eq("week_start", self.week_start.isoformat()).execute()

        log.info("✅ Phase 1 complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — CHAPTER OUTLINES (pre-plan all 21 chapters with suspense arcs)
    # ──────────────────────────────────────────────────────────────────────────

    def run_phase_2_chapter_outlines(self):
        log.info("=" * 60)
        log.info("📋  Phase 2: Chapter Outlines")
        log.info("=" * 60)

        articles = (
            supabase.table("articles")
            .select("*, characters(*)")
            .eq("week_start", self.week_start.isoformat())
            .eq("status", "character_gen")
            .execute()
            .data
        ) or []

        for article in articles:
            title      = article["title"]
            logline    = article["logline"]
            global_p   = article["global_prompt"]
            characters = article.get("characters", [])
            char_desc  = "\n".join(
                f"- {c['name']} ({c['gender']}, {c['age']}): {c['description']}"
                for c in characters
            )

            log.info("📖  Planning 21 chapters for: %s", title)

            outline_prompt = f"""You are a master manga writer. Plan 21 chapters for this manga series.

SERIES: {title}
LOGLINE: {logline}
OVERVIEW: {global_p}
CHARACTERS:
{char_desc}

Rules:
- Each chapter must end with a CLIFFHANGER or SUSPENSE HOOK that forces the reader to continue
- Chapter flow must be continuous — each chapter begins where the last ended
- Build tension progressively: chapters 1-7 = setup, 8-14 = escalation, 15-21 = climax/resolution
- Make it feel written by a human — raw emotions, unexpected twists, real dialogue moments
- Include 3 "visual scene prompts" per chapter for image generation (describe what's visually happening)

Return ONLY valid JSON array (no markdown):
[
  {{
    "chapter_number": 1,
    "title": "Evocative chapter title",
    "opening_hook": "How this chapter starts (1-2 sentences)",
    "summary": "What happens in this chapter (3-4 sentences)",
    "emotional_beat": "The core emotion of this chapter",
    "cliffhanger": "Exactly how this chapter ends to make reader desperate for next chapter",
    "scene_prompts": [
      "Detailed visual description of scene 1 (what characters are doing, setting, mood, lighting)",
      "Detailed visual description of scene 2",
      "Detailed visual description of scene 3"
    ]
  }}
]"""

            raw = gemini_generate(outline_prompt)
            if not raw:
                log.warning("  No outline response for %s", title)
                continue

            outlines = clean_json(raw)
            if not outlines or not isinstance(outlines, list):
                log.warning("  Bad outline JSON for %s", title)
                continue

            # Insert chapters with outlines stored in content as structured data
            # scheduled_date: all on same day (today) — publisher drip-feeds them
            base_date = self.week_start

            for outline in outlines[:21]:
                ch_num = safe_int(outline.get("chapter_number"), 1)
                # Spread publish dates: 3 chapters per day starting day 1
                day_offset   = (ch_num - 1) // 3
                publish_date = (base_date + timedelta(days=day_offset)).isoformat()

                # Store the outline JSON in content for Phase 3 to use
                content_json = json.dumps({
                    "opening_hook"  : outline.get("opening_hook", ""),
                    "summary"       : outline.get("summary", ""),
                    "emotional_beat": outline.get("emotional_beat", ""),
                    "cliffhanger"   : outline.get("cliffhanger", ""),
                    "scene_prompts" : outline.get("scene_prompts", []),
                    "status"        : "outline_only",  # will be replaced in Phase 3
                })

                try:
                    supabase.table("chapters").insert({
                        "article_id"    : article["id"],
                        "chapter_number": ch_num,
                        "title"         : str(outline.get("title", f"Chapter {ch_num}"))[:200],
                        "content"       : content_json,
                        "scheduled_date": publish_date,
                    }).execute()
                except Exception as e:
                    log.error("  Insert chapter %d error: %s", ch_num, e)

            supabase.table("articles").update({
                "status": "asset_gen"
            }).eq("id", article["id"]).execute()

            log.info("  ✓ 21 chapter outlines created for %s", title)
            time.sleep(2)

        supabase.table("pipeline_runs").update({
            "phase": "writing"
        }).eq("week_start", self.week_start.isoformat()).execute()

        log.info("✅ Phase 2 complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — WRITING (expand outlines into full human-like chapters)
    # ──────────────────────────────────────────────────────────────────────────

    def run_phase_3_writing(self):
        log.info("=" * 60)
        log.info("✍️   Phase 3: Writing Full Chapters")
        log.info("=" * 60)

        articles = (
            supabase.table("articles")
            .select("*, characters(*)")
            .eq("week_start", self.week_start.isoformat())
            .eq("status", "asset_gen")
            .execute()
            .data
        ) or []

        for article in articles:
            title    = article["title"]
            global_p = article["global_prompt"]
            chars    = article.get("characters", [])
            char_names = " & ".join(c["name"] for c in chars[:2])

            chapters = (
                supabase.table("chapters")
                .select("*")
                .eq("article_id", article["id"])
                .order("chapter_number")
                .execute()
                .data
            ) or []

            log.info("✍️  Writing %d chapters for: %s", len(chapters), title)

            previous_ending = ""

            for chapter in chapters:
                ch_num = chapter["chapter_number"]

                # Parse stored outline
                try:
                    outline = json.loads(chapter["content"])
                except Exception:
                    outline = {}

                if outline.get("status") != "outline_only":
                    log.info("  Ch.%d already written — skipping", ch_num)
                    continue

                opening  = outline.get("opening_hook", "")
                summary  = outline.get("summary", "")
                emotion  = outline.get("emotional_beat", "intensity")
                cliffhgr = outline.get("cliffhanger", "")

                prev_context = f"\nPREVIOUS CHAPTER ENDED: {previous_ending}" if previous_ending else ""

                write_prompt = f"""You are an acclaimed manga writer known for emotionally gripping, human storytelling.

SERIES: {title}
CHARACTERS: {char_names}
SERIES OVERVIEW: {global_p}
{prev_context}

Write Chapter {ch_num}: "{chapter['title']}"

OUTLINE:
- Opens with: {opening}
- Story beat: {summary}
- Core emotion: {emotion}
- Must end with: {cliffhanger}

WRITING RULES:
1. Write in present tense, manga panel-style prose (short punchy sentences, occasional single-word lines)
2. Include internal monologue in *italics* markers like *this*
3. Show emotion through physical reactions, not "she felt sad"
4. Dialogue must feel natural and character-specific
5. Minimum 400 words, maximum 600 words
6. The final paragraph MUST deliver the cliffhanger in a way that makes it impossible not to read the next chapter
7. NO chapter number headers — start immediately with the story

Write the chapter now:""".replace("{cliffhanger}", cliffhgr)

                raw = gemini_generate(write_prompt)
                if not raw or len(raw) < 100:
                    log.warning("  Ch.%d: bad response", ch_num)
                    raw = f"{opening}\n\n{summary}\n\n{cliffhgr}"

                # Store full content + preserve scene_prompts for image gen
                final_content = json.dumps({
                    "prose"         : raw[:3000],
                    "scene_prompts" : outline.get("scene_prompts", []),
                    "cliffhanger"   : cliffhgr,
                    "status"        : "written",
                })

                supabase.table("chapters").update({
                    "content": final_content
                }).eq("id", chapter["id"]).execute()

                previous_ending = cliffhgr
                log.info("  ✓ Ch.%d written (%d chars)", ch_num, len(raw))
                time.sleep(1)

            supabase.table("articles").update({
                "status": "writing"
            }).eq("id", article["id"]).execute()

        supabase.table("pipeline_runs").update({
            "phase": "image_gen"
        }).eq("week_start", self.week_start.isoformat()).execute()

        log.info("✅ Phase 3 complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — IMAGE GENERATION (precise prompts from scene_prompts)
    # ──────────────────────────────────────────────────────────────────────────

    def run_phase_4_images(self):
        log.info("=" * 60)
        log.info("🎨  Phase 4: Image Generation")
        log.info("=" * 60)

        articles = (
            supabase.table("articles")
            .select("*, characters(*)")
            .eq("week_start", self.week_start.isoformat())
            .eq("status", "writing")
            .execute()
            .data
        ) or []

        total_done   = 0
        total_failed = 0

        for article in articles:
            title  = article["title"]
            chars  = article.get("characters", [])
            char_prompts = [c.get("master_prompt", c["name"]) for c in chars[:2]]

            chapters = (
                supabase.table("chapters")
                .select("*")
                .eq("article_id", article["id"])
                .order("chapter_number")
                .execute()
                .data
            ) or []

            log.info("🎨  Generating images for: %s (%d chapters)", title, len(chapters))

            for chapter in chapters:
                ch_num = chapter["chapter_number"]

                try:
                    content = json.loads(chapter["content"])
                except Exception:
                    continue

                scene_prompts = content.get("scene_prompts", [])
                if not scene_prompts:
                    scene_prompts = [
                        "two characters in dramatic confrontation",
                        "intense emotional close-up",
                        "wide establishing shot of the scene"
                    ]

                for idx, scene in enumerate(scene_prompts[:3]):
                    # Check if image already exists
                    existing = (
                        supabase.table("images")
                        .select("id")
                        .eq("chapter_id", chapter["id"])
                        .eq("placement_index", idx + 1)
                        .execute()
                        .data
                    )
                    if existing:
                        continue

                    # Build precise prompt: character consistency + scene context
                    char_str   = ", ".join(char_prompts)
                    full_prompt = f"{char_str}, {scene[:200]}, {STYLE_POS}"

                    seed_str  = f"{article['id']}_{ch_num}_{idx}"
                    seed      = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                    filename  = f"{article['id']}_{ch_num}_{idx + 1}.png"

                    log.info("  🖼️  Ch.%d Scene %d — %s", ch_num, idx + 1, scene[:60])

                    image_bytes = hf_generate_image(full_prompt)

                    if not image_bytes:
                        log.warning("  ❌ Failed — inserting placeholder")
                        url = f"https://picsum.photos/seed/{seed}/512/512"
                        total_failed += 1
                    else:
                        url = upload_to_r2(image_bytes, filename)
                        if not url:
                            url = f"https://picsum.photos/seed/{seed}/512/512"
                            total_failed += 1
                        else:
                            total_done += 1

                    supabase.table("images").insert({
                        "chapter_id"     : chapter["id"],
                        "image_url"      : url,
                        "prompt"         : full_prompt[:500],
                        "seed"           : seed,
                        "placement_index": idx + 1,
                        "description"    : scene[:200],
                    }).execute()

                    time.sleep(3)  # be gentle with HF API

            supabase.table("articles").update({
                "status": "publishing"
            }).eq("id", article["id"]).execute()

        supabase.table("pipeline_runs").update({
            "phase"           : "publishing",
            "images_generated": total_done,
        }).eq("week_start", self.week_start.isoformat()).execute()

        log.info("📊  Images done: %d  failed: %d", total_done, total_failed)
        log.info("✅ Phase 4 complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5 — PUBLISHING
    # ──────────────────────────────────────────────────────────────────────────

    def run_phase_5_publishing(self):
        log.info("=" * 60)
        log.info("📢  Phase 5: Publishing")
        log.info("=" * 60)

        articles = (
            supabase.table("articles")
            .select("*")
            .eq("week_start", self.week_start.isoformat())
            .eq("status", "publishing")
            .execute()
            .data
        ) or []

        total = 0
        for article in articles:
            # Publish chapters scheduled for today
            today = date.today().isoformat()
            chapters = (
                supabase.table("chapters")
                .select("*, images(id, image_url)")
                .eq("article_id", article["id"])
                .eq("scheduled_date", today)
                .execute()
                .data
            ) or []

            for ch in chapters:
                # Validate: must have written content and at least 1 real image
                try:
                    content = json.loads(ch["content"])
                    prose = content.get("prose", "")
                except Exception:
                    prose = ch.get("content", "")

                if len(prose) < 100:
                    log.warning("  ⏭️  Ch.%d — no real content yet", ch["chapter_number"])
                    continue

                images = ch.get("images", [])
                real_images = [i for i in images if "picsum" not in i.get("image_url", "")]
                if not real_images:
                    log.warning("  ⏭️  Ch.%d — no real images yet", ch["chapter_number"])
                    continue

                supabase.table("chapters").update({
                    "published"   : True,
                    "published_at": datetime.utcnow().isoformat(),
                    "status"      : "published",
                }).eq("id", ch["id"]).execute()
                total += 1
                log.info("  ✅ Published Ch.%d of %s", ch["chapter_number"], article["title"])

            supabase.table("articles").update({"status": "completed"}).eq("id", article["id"]).execute()

        supabase.table("pipeline_runs").update({
            "phase"           : "completed",
            "phase_progress"  : 100,
            "completed_at"    : datetime.utcnow().isoformat(),
        }).eq("week_start", self.week_start.isoformat()).execute()

        log.info("📊  Total published today: %d", total)
        log.info("✅ Phase 5 complete!")

    # ──────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self):
        log.info("")
        log.info("█" * 60)
        log.info("  🎬 MangaForge AI — Full Pipeline")
        log.info("  Week: %s", self.week_start)
        log.info("█" * 60)
        self.run_phase_1_ideation()
        self.run_phase_2_chapter_outlines()
        self.run_phase_3_writing()
        self.run_phase_4_images()
        self.run_phase_5_publishing()
        log.info("")
        log.info("█" * 60)
        log.info("  ✨ Full pipeline complete!")
        log.info("█" * 60)


if __name__ == "__main__":
    pipeline = MangaForgePipeline()
    pipeline.run_full_pipeline()