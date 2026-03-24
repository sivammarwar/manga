"""
image_generator_hf.py
=====================
Cloud fallback image generator using HuggingFace Inference API.
Used by GitHub Actions when your M1 Mac is off.
Generates ~150 images/day for free via HF free tier.

Your M1 Mac + Stable Diffusion is the PRIMARY generator.
This script is the FALLBACK that runs in the cloud daily.
"""

import os
import io
import time
import hashlib
import logging
import requests
import boto3
from botocore.config import Config
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("image_generator_hf.log")],
)
log = logging.getLogger("hfgen")

# ─── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_API_TOKEN")
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"}

HF_MODELS = [
    "dreamlike-art/dreamlike-anime-1-0",
    "Linaqruf/anything-v3-better-vae",
    "hakurei/waifu-diffusion",
]

DAILY_LIMIT = 150
DELAY       = 4    # seconds between requests

STYLE_POS = "anime style, manga panel, detailed, vibrant colors, masterpiece, best quality"
STYLE_NEG = "lowres, bad anatomy, worst quality, blurry, deformed, ugly, text, watermark"

R2_ENDPOINT  = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS    = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET    = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET    = os.getenv("R2_BUCKET_NAME", "manga-images")
R2_PUBLIC    = os.getenv("R2_PUBLIC_URL")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


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
        r2.put_object(Bucket=R2_BUCKET, Key=key, Body=image_bytes,
                      ContentType="image/png", CacheControl="public, max-age=31536000")
        return f"{R2_PUBLIC}/{key}"
    except Exception as e:
        log.error("R2 upload error: %s", e)
        return None


def call_hf(prompt: str, model: str) -> bytes | None:
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": STYLE_NEG,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
        }
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=HEADERS, json=payload, timeout=120
            )
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 30)
                log.info("  ⏳ Model loading... %ds", wait)
                time.sleep(wait + 5)
                continue
            if resp.status_code == 429:
                log.warning("  ⚠️  Rate limit — waiting 60s")
                time.sleep(60)
                continue
            log.error("  HTTP %d: %s", resp.status_code, resp.text[:60])
            return None
        except Exception as e:
            log.error("  Attempt %d error: %s", attempt + 1, e)
            time.sleep(10)
    return None


def extract_scenes(text: str, count: int) -> list[str]:
    fallbacks = [
        "two characters in dramatic confrontation, detailed anime background",
        "intense emotional close-up of anime character",
        "wide shot of fantasy landscape with anime characters",
        "action scene with dynamic movement, manga style",
        "quiet emotional moment between characters at sunset",
    ]
    if not text or len(text) < 50:
        return fallbacks[:count]
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
    if not sentences:
        return fallbacks[:count]
    step = max(1, len(sentences) // count)
    scenes = [sentences[min(i * step, len(sentences) - 1)] for i in range(count)]
    while len(scenes) < count:
        scenes.append(fallbacks[len(scenes) % len(fallbacks)])
    return scenes[:count]


def run():
    log.info("=" * 55)
    log.info("☁️   HF IMAGE FALLBACK  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("    Daily limit: %d images", DAILY_LIMIT)
    log.info("=" * 55)

    result = (
        supabase.table("images")
        .select("id, chapter_id, placement_index")
        .like("image_url", "%picsum%")
        .limit(DAILY_LIMIT)
        .execute()
    )
    placeholders = result.data

    if not placeholders:
        log.info("✅ No placeholders left!")
        return

    log.info("Found %d placeholders to replace.\n", len(placeholders))

    chapter_groups: dict[str, list] = {}
    for img in placeholders:
        chapter_groups.setdefault(img["chapter_id"], []).append(img)

    done = 0
    failed = 0
    model_idx = 0

    for chapter_id, images in chapter_groups.items():
        try:
            chapter    = supabase.table("chapters").select("*").eq("id", chapter_id).single().execute().data
            article    = supabase.table("articles").select("*").eq("id", chapter["article_id"]).single().execute().data
            characters = supabase.table("characters").select("*").eq("article_id", chapter["article_id"]).execute().data
        except Exception as e:
            log.error("DB fetch error: %s", e)
            continue

        ch_num = chapter["chapter_number"]
        title  = article.get("title", "?")[:35]
        log.info("📖 %s — Chapter %d", title, ch_num)

        scenes = extract_scenes(chapter.get("content", ""), len(images))

        for i, img_rec in enumerate(images):
            scene  = scenes[min(i, len(scenes) - 1)]
            char_strs = [c.get("master_prompt", c["name"]) for c in characters[:2]]
            prompt = f"{', '.join(char_strs)}, {scene[:120]}, {STYLE_POS}"

            seed_str = f"{chapter['article_id']}_{ch_num}_{img_rec['placement_index']}"
            seed     = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            filename = f"{chapter['article_id']}_{ch_num}_{img_rec['placement_index']}.png"

            model = HF_MODELS[model_idx % len(HF_MODELS)]
            log.info("  🎨 Scene %d — %s", i + 1, scene[:50])

            image_bytes = call_hf(prompt, model)

            if not image_bytes:
                model_idx += 1
                image_bytes = call_hf(prompt, HF_MODELS[model_idx % len(HF_MODELS)])

            if not image_bytes:
                log.warning("  ❌ All models failed — skipping")
                failed += 1
                continue

            url = upload_to_r2(image_bytes, filename)
            if url:
                supabase.table("images").update({
                    "image_url": url,
                    "prompt"   : prompt[:500],
                    "seed"     : seed,
                }).eq("id", img_rec["id"]).execute()
                done += 1
                log.info("  ✅ %s", url[:65])
            else:
                failed += 1

            time.sleep(DELAY)

    log.info("")
    log.info("=" * 55)
    log.info("📊  Done: %d  Failed: %d", done, failed)
    log.info("=" * 55)


if __name__ == "__main__":
    run()