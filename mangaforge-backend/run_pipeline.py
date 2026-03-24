import os
import json
import time
import re
from datetime import datetime, timedelta

from google import genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

MODEL = "models/gemini-2.5-flash"
CATEGORIES = ['romantic', 'action', 'emotional', 'sad', 'spicy', 'horror', 'comedy', 'fantasy', 'mystery', 'scifi']

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

def generate_json(prompt):
    """Generate and parse JSON from Gemini"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            text = response.text
            # Extract JSON
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {str(e)[:100]}")
            time.sleep(2)
    return None

def to_int(value):
    """Safely convert to integer - CRITICAL for age field"""
    try:
        if value is None:
            return 20
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            # Remove decimal and any non-numeric except minus
            value = value.strip()
            # Handle "10.0" -> "10"
            if '.' in value:
                value = value.split('.')[0]
            return int(value)
        return 20
    except:
        return 20

print("=" * 60)
print("🎬 MangaForge AI Pipeline Starting")
print("=" * 60)

week_start = datetime.now().date().isoformat()
print(f"\n📅 Week Starting: {week_start}\n")

# Clear existing data
print("🧹 Clearing existing data...")
supabase.table('images').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
supabase.table('chapters').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
supabase.table('characters').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
supabase.table('articles').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
supabase.table('pipeline_runs').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
print("✓ Data cleared\n")

# Create pipeline record
supabase.table('pipeline_runs').insert({
    'week_start': week_start,
    'phase': 'ideation',
    'phase_progress': 0,
    'series_count': 10
}).execute()

# Generate series
for i, category in enumerate(CATEGORIES):
    print(f"[{i+1}/10] Generating {category.upper()} series...")
    
    prompt = f"""
    Create a unique manga premise for the {category} genre.
    Genre description: {CATEGORY_DESC[category]}
    
    Return ONLY a valid JSON object with no markdown:
    {{
        "title": "Series Title",
        "logline": "One sentence hook",
        "global_prompt": "Full 3-act story summary (short)",
        "characters": [
            {{
                "name": "Character Name",
                "age": 20,
                "gender": "male/female",
                "description": "Physical description",
                "personality": ["trait1", "trait2"]
            }}
        ]
    }}
    """
    
    data = generate_json(prompt)
    if not data:
        print(f"  ✗ Failed to generate for {category}")
        continue
    
    # Insert article
    try:
        article = supabase.table('articles').insert({
            'title': str(data.get('title', f'{category} Series'))[:150],
            'category': category,
            'logline': str(data.get('logline', ''))[:300],
            'global_prompt': str(data.get('global_prompt', ''))[:1000],
            'status': 'character_gen',
            'week_start': week_start
        }).execute()
        
        article_id = article.data[0]['id']
        print(f"  ✓ Article: {data.get('title')[:50]}")
        
        # Insert characters
        characters_inserted = 0
        for char in data.get('characters', [])[:3]:
            age = to_int(char.get('age'))
            print(f"    Age conversion: {char.get('age')} -> {age} (type: {type(age).__name__})")
            
            supabase.table('characters').insert({
                'article_id': article_id,
                'name': str(char.get('name', 'Unknown'))[:100],
                'age': age,
                'gender': str(char.get('gender', 'other'))[:20],
                'description': str(char.get('description', ''))[:300],
                'personality': [str(p)[:50] for p in char.get('personality', [])[:3]],
                'master_prompt': f"anime style, {char.get('name', 'character')}"
            }).execute()
            characters_inserted += 1
        
        print(f"  ✓ Characters: {characters_inserted}")
        
        # Update progress
        progress = ((i + 1) / 10) * 100
        supabase.table('pipeline_runs').update({
            'phase_progress': progress
        }).eq('week_start', week_start).execute()
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue
    
    time.sleep(2)

# Update pipeline status
supabase.table('pipeline_runs').update({
    'phase': 'completed',
    'phase_progress': 100,
    'completed_at': datetime.now().isoformat()
}).eq('week_start', week_start).execute()

print("\n" + "=" * 60)
print("✨ Pipeline Complete!")
print("=" * 60)

# Show results
result = supabase.table('articles').select('*').execute()
print(f"\n📊 Results:")
print(f"  Total articles: {len(result.data)}/10")
for a in result.data:
    print(f"    - {a['title']} ({a['category']})")
