import schedule
import time
from datetime import datetime
from pipeline import MangaForgePipeline

def run_weekly_pipeline():
    """Run the full pipeline at start of week"""
    print(f"\n{'='*50}")
    print(f"Starting weekly pipeline at {datetime.now()}")
    print(f"{'='*50}")
    
    pipeline = MangaForgePipeline()
    pipeline.run_full_pipeline()

# Schedule to run every Monday at 6 AM
schedule.every().monday.at("06:00").do(run_weekly_pipeline)

print("📅 Scheduler started. Waiting for Monday 6:00 AM...")
print("To run manually: python pipeline.py")

# Keep running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
