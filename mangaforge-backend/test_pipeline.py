import os
import json
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict

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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_int(value) -> int:
    """Safely convert any value to integer"""
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
    """Extract and parse JSON from response"""
    if not text:
        return None
    
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except:
            return None
    return None


def generate_json(prompt: str) -> Optional[Dict]:
    """Call Gemini API"""
    try:
        print("    → Calling Gemini API...", end=" ")
        response = client.models.generate_content(model=MODEL, contents=prompt)
        data = clean_json(response.text)
        
        if data:
            print("✓")
            time.sleep(15)  # Wait to avoid quota
            return data
        else:
            print("✗ No JSON found")
            return None
    
    except Exception as e:
        print(f"✗ {str(e)[:60]}")
        if "429" in str(e) or "quota" in str(e).lower():
            print("    ⚠️  Quota hit - waiting 30 seconds...")
            time.sleep(30)
        return None


# ============================================================================
# TEST PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("🧪 MANGAFORGE TEST PIPELINE (1 Category)")
print("=" * 70)

week_start = datetime.now().date().isoformat()
print(f"\n📅 Week: {week_start}")

# Test connection
print("\n🔌 Testing connections...")
try:
    print("  → Supabase...", end=" ")
    result = supabase.table('articles').select('count').execute()
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    exit(1)

try:
    print("  → Gemini API...", end=" ")
    response = client.models.generate_content(
        model=MODEL,
        contents="Say 'Hello' only, nothing else."
    )
    print("✓\n")
except Exception as e:
    print(f"✗ {e}")
    exit(1)

# Generate 1 test series
print("🎨 Generating ACTION series (test)...\n")

prompt = """
Create a manga premise for the ACTION genre.

Return ONLY JSON:
{
    "title": "Test Action Series",
    "logline": "A test premise",
    "global_prompt": "Test story",
    "characters": [
        {
            "name": "Hero",
            "age": 25,
            "gender": "male",
            "description": "Brave warrior",
            "personality": ["brave", "strong"]
        }
    ]
}
"""

print("Step 1: Generate series premise")
data = generate_json(prompt)

if not data:
    print("✗ Failed to generate series")
    exit(1)

print(f"  ✓ Title: {data.get('title')}")

# Insert article
print("\nStep 2: Insert article into Supabase")
try:
    article_resp = supabase.table('articles').insert({
        'title': str(data.get('title', 'Test Series'))[:150],
        'category': 'action',
        'logline': str(data.get('logline', ''))[:300],
        'global_prompt': str(data.get('global_prompt', ''))[:1000],
        'status': 'character_gen',
        'week_start': week_start
    }).execute()
    
    article_id = article_resp.data[0]['id']
    print(f"  ✓ Article ID: {article_id}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Insert characters
print("\nStep 3: Insert characters")
char_count = 0
for char in data.get('characters', [])[:2]:
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
        print(f"  ✓ {char.get('name')} (age: {age})")
    except Exception as e:
        print(f"  ✗ {e}")

# Create chapters
print("\nStep 4: Create 3 test chapters")
for ch_num in range(1, 4):
    try:
        ch_resp = supabase.table('chapters').insert({
            'article_id': article_id,
            'chapter_number': ch_num,
            'title': f"Chapter {ch_num}",
            'content': f"Test content for chapter {ch_num}",
            'scheduled_date': (datetime.now().date() + timedelta(days=ch_num)).isoformat()
        }).execute()
        
        ch_id = ch_resp.data[0]['id']
        
        # Add 3 placeholder images
        for img_idx in range(3):
            supabase.table('images').insert({
                'chapter_id': ch_id,
                'image_url': f"https://picsum.photos/seed/test{ch_num}{img_idx}/800/500",
                'prompt': f"Scene {img_idx+1}",
                'seed': 1000 + ch_num * 100 + img_idx,
                'description': f"Test image {img_idx+1}",
                'placement_index': img_idx + 1
            }).execute()
        
        print(f"  ✓ Chapter {ch_num} + 3 images")
    except Exception as e:
        print(f"  ✗ Chapter {ch_num}: {e}")

# Verify data
print("\n" + "=" * 70)
print("✅ TEST COMPLETE - Verifying data...\n")

try:
    articles = supabase.table('articles').select('*').execute()
    print(f"Articles in DB: {len(articles.data)}")
    for a in articles.data:
        print(f"  - {a['title']} ({a['category']})")
    
    chars = supabase.table('characters').select('*').execute()
    print(f"\nCharacters: {len(chars.data)}")
    for c in chars.data:
        print(f"  - {c['name']} (age: {c['age']})")
    
    chapters = supabase.table('chapters').select('*').execute()
    print(f"\nChapters: {len(chapters.data)}")
    
    images = supabase.table('images').select('*').execute()
    print(f"Images: {len(images.data)}")
    
    print("\n" + "=" * 70)
    print("✨ Everything works! Ready to run full pipeline.")
    print("=" * 70 + "\n")

except Exception as e:
    print(f"✗ Verification error: {e}")