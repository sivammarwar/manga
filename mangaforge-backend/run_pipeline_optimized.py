import os
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from google import genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

MODEL = "models/gemini-2.5-flash"

CATEGORIES = [
    'romantic', 'action', 'emotional', 'sad', 'spicy',
    'horror', 'comedy', 'fantasy', 'mystery', 'scifi'
]

CATEGORY_DESC = {
    'romantic': 'Heartwarming love stories with emotional depth',
    'action': 'High-octane battles and heroic adventures',
    'emotional': 'Deep narratives exploring human emotions',
    'sad': 'Bittersweet tales of loss and longing',
    'spicy': 'Passionate, mature romance with heat',
    'horror': 'Dark, chilling tales of terror',
    'comedy': 'Hilarious adventures and witty humor',
    'fantasy': 'Epic quests in magical worlds',
    'mystery': 'Puzzles, clues, and shocking revelations',
    'scifi': 'Futuristic worlds and cosmic adventures'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_int(value) -> int:
    """Safely convert any value to integer (handles "10.0" → 10)"""
    try:
        if value is None:
            return 20
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if '.' in value:
                value = value.split('.')[0]
            return int(value)
        return 20
    except:
        return 20


def clean_json(text: str) -> Optional[Dict]:
    """Extract and parse JSON from response text"""
    if not text:
        return None
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object boundaries
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def generate_json(prompt: str, delay_after: int = 15) -> Optional[Dict]:
    """
    Call Gemini API to generate JSON with retry logic.
    
    Args:
        prompt: The prompt to send
        delay_after: Seconds to wait after successful response (default 15 to avoid quota)
    
    Returns:
        Parsed JSON dict or None if failed
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"    [Attempt {attempt + 1}/{max_retries}] Calling Gemini API...", end=" ")
            response = client.models.generate_content(model=MODEL, contents=prompt)
            
            data = clean_json(response.text)
            if data:
                print("✓ Success")
                # CRITICAL: Wait after successful response to avoid quota limit
                time.sleep(delay_after)
                return data
            else:
                print("✗ No valid JSON in response")
        
        except Exception as e:
            error_str = str(e)
            print(f"✗ Error: {error_str[:80]}")
            
            # If it's a quota error, wait longer before retry
            if "429" in error_str or "quota" in error_str.lower():
                print(f"    ⚠️  Quota limit hit! Waiting 30 seconds before retry...")
                time.sleep(30)
            else:
                time.sleep(5)
        
        if attempt < max_retries - 1:
            time.sleep(5)
    
    return None


# ============================================================================
# PIPELINE PHASES
# ============================================================================

class MangaForgePipeline:
    def __init__(self):
        self.week_start = datetime.now().date()
        self.week_start_str = self.week_start.isoformat()
        self.stats = {
            'articles_created': 0,
            'characters_created': 0,
            'chapters_created': 0,
            'images_created': 0,
            'errors': []
        }
    
    def clear_database(self):
        """Clear all existing data"""
        print("\n🧹 Clearing existing data...")
        try:
            supabase.table('images').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            supabase.table('chapters').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            supabase.table('characters').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            supabase.table('articles').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            supabase.table('pipeline_runs').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            print("✓ Database cleared")
        except Exception as e:
            print(f"✗ Error clearing database: {e}")
    
    def phase_1_ideation(self):
        """Generate 10 unique series with characters"""
        print("\n" + "=" * 70)
        print("🚀 PHASE 1: IDEATION (Generate 10 Series)")
        print("=" * 70)
        
        # Create pipeline record
        try:
            supabase.table('pipeline_runs').insert({
                'week_start': self.week_start_str,
                'phase': 'ideation',
                'phase_progress': 0,
                'series_count': 10,
                'started_at': datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"✗ Error creating pipeline record: {e}")
        
        for i, category in enumerate(CATEGORIES):
            print(f"\n[{i+1}/10] 🎨 Generating {category.upper()} series...")
            
            prompt = f"""
Create a UNIQUE manga premise for the {category} genre.
Genre description: {CATEGORY_DESC[category]}

IMPORTANT: Create original characters with realistic ages (18-35).

Return ONLY a valid JSON object with NO markdown formatting:
{{
    "title": "Creative Series Title (not generic)",
    "logline": "One compelling sentence hook",
    "global_prompt": "3-act story summary (2-3 sentences)",
    "characters": [
        {{
            "name": "Character Name",
            "age": 25,
            "gender": "male/female",
            "description": "Physical and personality description",
            "personality": ["trait1", "trait2", "trait3"]
        }}
    ]
}}

Ensure all character ages are numbers between 18-35.
"""
            
            data = generate_json(prompt, delay_after=15)
            
            if not data:
                self.stats['errors'].append(f"Failed to generate {category}")
                print(f"  ✗ Failed to generate JSON for {category}")
                continue
            
            try:
                # Insert article
                article_resp = supabase.table('articles').insert({
                    'title': str(data.get('title', f'{category.title()} Series'))[:150],
                    'category': category,
                    'logline': str(data.get('logline', ''))[:300],
                    'global_prompt': str(data.get('global_prompt', ''))[:1000],
                    'status': 'character_gen',
                    'week_start': self.week_start_str
                }).execute()
                
                article_id = article_resp.data[0]['id']
                self.stats['articles_created'] += 1
                print(f"  ✓ Article created: {data.get('title', 'Unknown')[:50]}")
                
                # Insert characters
                char_count = 0
                for char in data.get('characters', [])[:3]:  # Max 3 chars per series
                    try:
                        age = to_int(char.get('age'))
                        
                        supabase.table('characters').insert({
                            'article_id': article_id,
                            'name': str(char.get('name', 'Unknown'))[:100],
                            'age': age,
                            'gender': str(char.get('gender', 'other'))[:20],
                            'description': str(char.get('description', ''))[:300],
                            'personality': [str(p)[:50] for p in char.get('personality', [])[:3]],
                            'master_prompt': f"anime style, {char.get('name', 'character')}"
                        }).execute()
                        char_count += 1
                        self.stats['characters_created'] += 1
                    
                    except Exception as e:
                        self.stats['errors'].append(f"Character insert error: {e}")
                        continue
                
                print(f"  ✓ Characters created: {char_count}/3")
                
                # Update progress
                progress = ((i + 1) / 10) * 100
                supabase.table('pipeline_runs').update({
                    'phase_progress': progress
                }).eq('week_start', self.week_start_str).execute()
            
            except Exception as e:
                self.stats['errors'].append(f"Article creation error: {e}")
                print(f"  ✗ Error creating article: {e}")
        
        # Update phase complete
        supabase.table('pipeline_runs').update({
            'phase': 'character_gen'
        }).eq('week_start', self.week_start_str).execute()
        
        print("\n✅ Phase 1 Complete!")
    
    def phase_2_asset_generation(self):
        """Create placeholder images for all chapters"""
        print("\n" + "=" * 70)
        print("🖼️  PHASE 2: ASSET GENERATION (Create Placeholders)")
        print("=" * 70)
        
        articles = supabase.table('articles').select('*').eq('status', 'character_gen').execute()
        print(f"Found {len(articles.data)} articles")
        
        for article in articles.data:
            article_id = article['id']
            title = article['title'][:40]
            
            print(f"\n📖 {title}...")
            
            # Get or create chapters
            chapters_resp = supabase.table('chapters').select('*').eq('article_id', article_id).execute()
            
            if not chapters_resp.data:
                print(f"  Creating 21 chapters...")
                for ch_num in range(1, 22):
                    scheduled = self.week_start + timedelta(days=3 + ((ch_num - 1) // 3))
                    
                    supabase.table('chapters').insert({
                        'article_id': article_id,
                        'chapter_number': ch_num,
                        'title': f"Chapter {ch_num}",
                        'content': f"Chapter {ch_num} content pending...",
                        'scheduled_date': scheduled.isoformat()
                    }).execute()
                    self.stats['chapters_created'] += 1
                
                chapters_resp = supabase.table('chapters').select('*').eq('article_id', article_id).execute()
            
            # Create placeholder images for each chapter
            for ch in chapters_resp.data:
                chapter_id = ch['id']
                ch_num = ch['chapter_number']
                
                for img_idx in range(5):  # 5 images per chapter
                    try:
                        supabase.table('images').insert({
                            'chapter_id': chapter_id,
                            'image_url': f"https://picsum.photos/seed/{chapter_id}_{img_idx}/800/500",
                            'prompt': f"Scene {img_idx + 1} from chapter {ch_num}",
                            'seed': int(chapter_id.replace('-', '')[:8], 16) + img_idx,
                            'description': f"Visual scene {img_idx + 1}",
                            'placement_index': img_idx + 1
                        }).execute()
                        self.stats['images_created'] += 1
                    except Exception as e:
                        self.stats['errors'].append(f"Image creation error: {e}")
            
            print(f"  ✓ Created 21 chapters × 5 images = 105 assets")
        
        # Update article status
        for article in articles.data:
            supabase.table('articles').update({'status': 'writing'}).eq('id', article['id']).execute()
        
        supabase.table('pipeline_runs').update({
            'phase': 'writing'
        }).eq('week_start', self.week_start_str).execute()
        
        print("\n✅ Phase 2 Complete!")
    
    def phase_3_writing(self):
        """Generate chapter content"""
        print("\n" + "=" * 70)
        print("✍️  PHASE 3: WRITING (Generate Chapter Content)")
        print("=" * 70)
        
        articles = supabase.table('articles').select('*').eq('status', 'writing').execute()
        print(f"Found {len(articles.data)} articles to write\n")
        
        for article in articles.data:
            article_id = article['id']
            title = article['title'][:40]
            global_prompt = article['global_prompt']
            
            print(f"📝 {title}...")
            
            chapters = supabase.table('chapters').select('*').eq('article_id', article_id).execute()
            
            for ch in chapters.data:
                chapter_id = ch['id']
                ch_num = ch['chapter_number']
                
                write_prompt = f"""
Write a SHORT manga chapter (80-100 words) for: "{article['title']}"

Series summary: {global_prompt}
Chapter {ch_num} focus: {ch['content'][:100] if ch['content'] else 'Main story progression'}

Write engaging, narrative-driven content suitable for manga format.
Use natural dialogue and action descriptions.
"""
                
                content = generate_json(write_prompt, delay_after=15)
                
                if content and isinstance(content, dict):
                    text_content = content.get('content', content.get('text', str(content)[:500]))
                elif isinstance(content, str):
                    text_content = content[:500]
                else:
                    text_content = f"Chapter {ch_num} content..."
                
                try:
                    supabase.table('chapters').update({
                        'content': text_content[:1000]
                    }).eq('id', chapter_id).execute()
                except Exception as e:
                    self.stats['errors'].append(f"Chapter update error: {e}")
            
            print(f"  ✓ Wrote 21 chapters")
        
        # Update status
        for article in articles.data:
            supabase.table('articles').update({'status': 'publishing'}).eq('id', article['id']).execute()
        
        supabase.table('pipeline_runs').update({
            'phase': 'publishing'
        }).eq('week_start', self.week_start_str).execute()
        
        print("\n✅ Phase 3 Complete!")
    
    def phase_4_publishing(self):
        """Mark all as ready for publishing"""
        print("\n" + "=" * 70)
        print("📢 PHASE 4: PUBLISHING (Mark Complete)")
        print("=" * 70)
        
        articles = supabase.table('articles').select('*').eq('status', 'publishing').execute()
        
        for article in articles.data:
            supabase.table('articles').update({'status': 'completed'}).eq('id', article['id']).execute()
        
        supabase.table('pipeline_runs').update({
            'phase': 'completed',
            'phase_progress': 100,
            'completed_at': datetime.now().isoformat()
        }).eq('week_start', self.week_start_str).execute()
        
        print(f"✓ Marked {len(articles.data)} articles as completed")
        print("\n✅ Phase 4 Complete!")
    
    def run(self):
        """Execute full pipeline"""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 15 + "🎬 MANGAFORGE AI PIPELINE" + " " * 29 + "║")
        print("╚" + "=" * 68 + "╝")
        
        start_time = time.time()
        
        try:
            self.clear_database()
            self.phase_1_ideation()
            self.phase_2_asset_generation()
            self.phase_3_writing()
            self.phase_4_publishing()
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Pipeline error: {e}")
        
        # Print final stats
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("📊 FINAL STATISTICS")
        print("=" * 70)
        print(f"✓ Articles created:  {self.stats['articles_created']}/10")
        print(f"✓ Characters:        {self.stats['characters_created']}/30")
        print(f"✓ Chapters:          {self.stats['chapters_created']}/210")
        print(f"✓ Images:            {self.stats['images_created']}/1050")
        print(f"✓ Total time:        {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for err in self.stats['errors'][:5]:
                print(f"   - {err}")
        
        print("\n" + "=" * 70)
        print("✨ Pipeline execution complete! Check Supabase dashboard for data.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    pipeline = MangaForgePipeline()
    pipeline.run()