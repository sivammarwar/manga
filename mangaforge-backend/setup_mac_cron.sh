#!/bin/bash
# setup_mac_cron.sh
# =================
# Sets up local cron jobs on your M1 Mac for:
#   1. Daily image generation (10 PM every night)
#   2. Daily publishing (8 AM every day)
#
# Run ONCE: bash setup_mac_cron.sh

BACKEND_DIR="/Users/shivamkumarsingh/Documents/AI ANIME/mangaforge-backend"
PYTHON="$BACKEND_DIR/venv/bin/python"
LOG_DIR="$BACKEND_DIR/logs"

mkdir -p "$LOG_DIR"

# Write crontab (preserves any existing entries)
(crontab -l 2>/dev/null; cat <<EOF

# ── MangaForge AI ────────────────────────────────────────────
# Image generation: every night at 10 PM (50 images, ~12 min on M1)
0 22 * * * cd "$BACKEND_DIR" && "$PYTHON" image_generator.py >> "$LOG_DIR/imggen_\$(date +\%Y\%m\%d).log" 2>&1

# Publisher: every morning at 8 AM
0 8 * * * cd "$BACKEND_DIR" && "$PYTHON" publisher.py >> "$LOG_DIR/publisher_\$(date +\%Y\%m\%d).log" 2>&1
# ─────────────────────────────────────────────────────────────
EOF
) | crontab -

echo "✅ Cron jobs installed!"
echo ""
echo "Verify with:  crontab -l"
echo ""
echo "Your jobs:"
echo "  10:00 PM daily → image_generator.py  (M1 SD generation)"
echo "   8:00 AM daily → publisher.py         (mark chapters live)"