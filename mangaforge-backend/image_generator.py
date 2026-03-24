"""
image_generator.py
==================
Generates anime images using Stable Diffusion on M1 Mac (Metal/MPS)
and uploads them directly to Cloudflare R2.

Run manually:   python image_generator.py
Run daily:      Called by cron / GitHub Actions (50 images per run)

Requirements:
    pip install diffusers transformers accelerate torch Pillow
                boto3 supabase python-dotenv safetensors
"""

import os
import io
import time
import hashlib
import logging
import boto3
from botocore.config import Config
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("image_generator.log"),
    ],
)
log = logging.getLogger("imggen")

# ─── Config ───────────────────────────────────────────────────────────────────

# Stable Diffusion model (downloads once ~2GB, cached in ~/.cache/huggingface)
SD_MODEL = "dreamlike-art/dreamlike-anime-1-0"

# M1 Metal GPU
DEVICE = "mps"
DTYPE  = torch.float16

# Image settings
IMG_W    = 512
IMG_H    = 512
STEPS    = 25      # 25 = good quality, ~15 sec on M1
CFG      = 7.5
BATCH    = 50      # images to generate per daily run (stay manageable)
DELAY    = 1.0     # seconds between images (let M1 breathe)

STYLE_POSITIVE = (
    "anime style, manga panel, detailed background, "
    "vibrant colors, masterpiece, best quality, 8k, sharp focus"
)
STYLE_NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra limbs, "
    "worst quality, blurry, deformed, ugly, text, watermark, "
    "western style, realistic, 3d render"
)

# Cloudflare R2 (S3-compatible)
R2_ENDPOINT    = os.getenv("R2_ENDPOINT_URL")       # https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY  = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET      = os.getenv("R2_BUCKET_NAME", "manga-images")
R2_PUBLIC_URL  = os.getenv("R2_PUBLIC_URL")         # https://pub-xxx.r2.dev  (enable in CF dashboard)

# Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ─── Cloudflare R2 Client ─────────────────────────────────────────────────────

def get_r2_client():
    """Return a boto3 S3 client pointed at Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_to_r2(image: Image.Image, filename: str) -> str | None:
    """
    Convert PIL image → PNG bytes → upload to Cloudflare R2.
    Returns public CDN URL or None on failure.
    """
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        image_bytes = buf.read()

        r2 = get_r2_client()
        object_key = f"chapters/{filename}"

        r2.put_object(
            Bucket=R2_BUCKET,
            Key=object_key,
            Body=image_bytes,
            ContentType="image/png",
            CacheControl="public, max-age=31536000",  # 1 year CDN cache
        )

        public_url = f"{R2_PUBLIC_URL}/{object_key}"
        log.info("  ☁️  R2 upload OK → %s", public_url[:70])
        return public_url

    except Exception as e:
        log.error("  ❌ R2 upload failed: %s", e)
        return None


# ─── Stable Diffusion on M1 ───────────────────────────────────────────────────

class M1AnimeGenerator:
    """
    Wraps a Stable Diffusion pipeline running on Apple M1 Metal GPU.
    Model is lazy-loaded on first call and reused for all subsequent images.
    """

    def __init__(self):
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return

        log.info("🔄 Loading SD model (first run downloads ~2GB, then cached)...")
        log.info("   Model: %s", SD_MODEL)

        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Fast DPM++ scheduler — best quality/speed tradeoff
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()   # crucial for M1 — saves GPU RAM

        self.pipe = pipe
        log.info("✅ Model loaded on M1 Metal GPU")

    def build_prompt(self, scene: str, characters: list[dict]) -> tuple[str, str]:
        """
        Build positive + negative prompts.
        Injects character master_prompt for consistency across chapters.
        """
        char_parts = []
        for c in characters[:2]:
            master = c.get("master_prompt") or (
                f"anime {c.get('gender', 'character')}, "
                f"{c.get('description', '')[:80]}"
            )
            char_parts.append(master.strip())

        char_str = ", ".join(char_parts)
        scene_str = scene[:150].strip()

        positive = f"{char_str}, {scene_str}, {STYLE_POSITIVE}"
        return positive, STYLE_NEGATIVE

    def generate(self, prompt: str, negative: str, seed: int) -> Image.Image | None:
        """Generate a single image. Returns PIL Image or None."""
        self._load()

        generator = torch.Generator(DEVICE).manual_seed(seed)

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                width=IMG_W,
                height=IMG_H,
                generator=generator,
            )
            return result.images[0]

        except Exception as e:
            log.error("  ❌ SD generation error: %s", e)
            return None


# ─── Scene Extractor ──────────────────────────────────────────────────────────

def extract_scenes(chapter_text: str, count: int = 5) -> list[str]:
    """
    Split chapter text into `count` visual scene descriptions for image prompts.
    Falls back to generic scenes if text is a placeholder.
    """
    fallbacks = [
        "two characters facing each other dramatically, detailed background",
        "intense close-up of character with emotional expression",
        "wide establishing shot of fantasy landscape with characters",
        "action scene with dynamic movement and motion blur",
        "quiet emotional moment between two characters at sunset",
    ]

    if not chapter_text or chapter_text.startswith("Chapter") or len(chapter_text) < 50:
        return fallbacks[:count]

    # Split by sentence, filter meaningful ones
    raw = chapter_text.replace("\n", " ").replace("  ", " ")
    sentences = [s.strip() for s in raw.split(".") if len(s.strip()) > 30]

    if len(sentences) < 2:
        return fallbacks[:count]

    # Pick evenly-spaced sentences as scene anchors
    step = max(1, len(sentences) // count)
    scenes = [sentences[min(i * step, len(sentences) - 1)] for i in range(count)]

    # Pad with fallbacks if needed
    while len(scenes) < count:
        scenes.append(fallbacks[len(scenes) % len(fallbacks)])

    return scenes[:count]


# ─── Core Pipeline Phase ──────────────────────────────────────────────────────

def run_image_phase(batch_size: int = BATCH):
    """
    Main entry point. Finds chapters with placeholder images in Supabase,
    generates real anime images, uploads to R2, and updates DB.

    Designed to run daily — picks up exactly where it left off.
    """
    log.info("")
    log.info("=" * 60)
    log.info("🖼️  MANGAFORGE IMAGE GENERATION")
    log.info("   Date: %s  |  Batch: %d images", datetime.now().strftime("%Y-%m-%d %H:%M"), batch_size)
    log.info("=" * 60)

    # ── 1. Find placeholder images (picsum.photos = not yet generated) ──
    try:
        result = (
            supabase.table("images")
            .select("id, chapter_id, placement_index, prompt")
            .like("image_url", "%picsum%")
            .limit(batch_size)
            .execute()
        )
        placeholder_images = result.data
    except Exception as e:
        log.error("❌ Supabase query failed: %s", e)
        return

    if not placeholder_images:
        log.info("✅ No placeholder images remaining — all generated!")
        return

    log.info("Found %d placeholder images to replace today.", len(placeholder_images))

    # ── 2. Group placeholders by chapter ──
    chapter_groups: dict[str, list[dict]] = {}
    for img in placeholder_images:
        cid = img["chapter_id"]
        chapter_groups.setdefault(cid, []).append(img)

    # ── 3. Generate per chapter ──
    generator    = M1AnimeGenerator()
    total_done   = 0
    total_failed = 0

    for chapter_id, images in chapter_groups.items():

        # Fetch chapter + article + characters
        try:
            chapter = (
                supabase.table("chapters")
                .select("*")
                .eq("id", chapter_id)
                .single()
                .execute()
                .data
            )
            article = (
                supabase.table("articles")
                .select("*")
                .eq("id", chapter["article_id"])
                .single()
                .execute()
                .data
            )
            characters = (
                supabase.table("characters")
                .select("*")
                .eq("article_id", chapter["article_id"])
                .execute()
                .data
            )
        except Exception as e:
            log.error("  ❌ DB fetch error for chapter %s: %s", chapter_id, e)
            continue

        ch_num    = chapter["chapter_number"]
        art_id    = chapter["article_id"]
        art_title = article.get("title", "unknown")[:35]

        log.info("")
        log.info("📖  %s — Chapter %d (%d images)", art_title, ch_num, len(images))

        scenes = extract_scenes(chapter.get("content", ""), len(images))

        for i, img_record in enumerate(images):
            placement = img_record["placement_index"]
            scene     = scenes[min(i, len(scenes) - 1)]

            # Deterministic seed = same chapter + placement always produces same image
            seed_str = f"{art_id}_{ch_num}_{placement}"
            seed     = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

            positive, negative = generator.build_prompt(scene, characters)
            filename = f"{art_id}_{ch_num}_{placement}.png"

            log.info("  🎨 Scene %d/%d  seed=%d  %s", i + 1, len(images), seed, scene[:55])

            image = generator.generate(positive, negative, seed)

            if image is None:
                total_failed += 1
                log.warning("  ⚠️  Generation failed — keeping placeholder")
                continue

            url = upload_to_r2(image, filename)

            if url:
                # Update the images row in Supabase
                try:
                    supabase.table("images").update({
                        "image_url": url,
                        "prompt"   : positive[:500],
                        "seed"     : seed,
                    }).eq("id", img_record["id"]).execute()
                    total_done += 1
                except Exception as e:
                    log.error("  ❌ DB update failed: %s", e)
            else:
                total_failed += 1

            time.sleep(DELAY)

    # ── 4. Summary ──
    log.info("")
    log.info("=" * 60)
    log.info("📊 DONE  ✅ %d generated   ❌ %d failed", total_done, total_failed)

    remaining = (
        supabase.table("images")
        .select("id", count="exact")
        .like("image_url", "%picsum%")
        .execute()
        .count
    )
    log.info("   Placeholders remaining in DB: %d", remaining or 0)
    if remaining and remaining > 0:
        days_left = (remaining + batch_size - 1) // batch_size
        log.info("   Estimated days to complete: %d", days_left)
    log.info("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_image_phase(batch_size=BATCH)