from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import traceback
from datetime import date
from urllib.parse import quote
from typing import Optional

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI(debug=True)

# Dil eşleştirme sözlüğü
LANGUAGE_MAPPING = {
    "en": "English",
    "tr": "Turkish",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "zh": "Chinese",
    "it": "Italian"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FormData(BaseModel):
    movies: str
    music: str
    brands: str
    gender: str
    language: Optional[str] = "en"  # Varsayılan İngilizce
    variation: Optional[int] = 0

# ✅ CulturalMap için AI fonksiyonu
def generate_cultural_map_insights(countries: list[str], language: str = "en") -> dict:
    if not countries:
        return {}

    target_language = LANGUAGE_MAPPING.get(language, "English")
    
    prompt = f"""
    CRITICAL INSTRUCTION: You MUST respond ENTIRELY in {target_language} language.
    ALL cultural insights and recommendations must be in {target_language}.
    
    For each of the following countries, return a JSON array of objects.
    Each object should include:
    - country (string) - country name can be in original language
    - culturalInsight (1-2 sentences about the culture) - MUST be in {target_language}
    - recommendation (a film, artist, or brand that represents it) - MUST be in {target_language}

    Countries: {', '.join(countries)}

    FINAL REMINDER: All descriptions and recommendations must be in {target_language}.
    Only respond with a valid JSON list.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )

    content = response.choices[0].message.content

    if not content:
        print("⚠️ GPT returned empty cultural map content")
        return {}

    try:
        parsed = json.loads(content)
        return {item["country"]: item for item in parsed if "country" in item}
    except Exception as e:
        print("❌ Failed to parse cultural map response:", e)
        return {}

# Qloo autocomplete
def autocomplete_entity(query: str, entity_type: str = "artist") -> Optional[str]:
    base_url = os.getenv("QLOO_API_URL")
    key = os.getenv("QLOO_API_KEY")
    safe_query = quote(query)
    url = f"{base_url}/v1/autocomplete?query={safe_query}"
    headers = {"x-api-key": key}

    response = requests.get(url, headers=headers)
    print(f"🔵 Autocomplete [{query}] → {response.status_code}:\n", response.text)

    if response.status_code == 200:
        results = response.json().get("results", [])
        for r in results:
            if entity_type in r.get("type", "").lower():
                return r.get("id", "")
    else:
        print(f"⚠️ Qloo Autocomplete fallback activated for: {query}")
    return None

# Qloo trending
def get_qloo_trending(entity_id: Optional[str], entity_type: str = "artist") -> list:
    if not entity_id:
        return []

    base_url = os.getenv("QLOO_API_URL")
    key = os.getenv("QLOO_API_KEY")

    today = date.today()
    start_date = f"{today.year}-01-01"
    end_date = today.isoformat()

    url = (
        f"{base_url}/v2/trending?"
        f"filter.start_date={start_date}&"
        f"filter.end_date={end_date}&"
        f"filter.type=urn:entity:{entity_type}&"
        f"signal.interests.entities={entity_id}"
    )

    headers = {"x-api-key": key}
    response = requests.get(url, headers=headers)
    print("🟣 Trending response:", response.status_code, response.text)

    if response.status_code == 200:
        data = response.json()
        items = data.get("results", [])
        return [i.get("name", "Unknown") for i in items if "name" in i]
    return []

# GPT persona oluştur
def generate_persona_from_taste(movies: str, music: str, brands: str, gender: str, qloo_suggestions: list, language: str = "en", variation_seed: int = 0):
    target_language = LANGUAGE_MAPPING.get(language, "English")
    
    prompt = f"""
    CRITICAL INSTRUCTION: You MUST respond ENTIRELY in {target_language} language. 
    EVERY SINGLE TEXT FIELD must be in {target_language}, including:
    - personaName
    - description  
    - traits (all 3 items)
    - insights.likelyInterests
    - insights.likelyBehaviors
    - therapySuggestion.summary
    - therapySuggestion.recommendation
    - therapySuggestion.dailyTip
    - archetype.name
    - archetype.description
    - culturalDNAScore (region names as keys - MUST be in {target_language})

    User preferences:
    - Favorite Movies: {movies}
    - Favorite Music: {music}
    - Favorite Brands: {brands}
    - Gender: {gender}
    - Qloo cultural suggestions: {qloo_suggestions if qloo_suggestions else "None available"}
    - Variation seed: {variation_seed}

    Return a JSON with:
    - personaName (string) - MUST be in {target_language}
    - description (1-2 sentence string) - MUST be in {target_language}
    - traits (list of 3 strings) - MUST be in {target_language}
    - insights (object with:
        likelyInterests (string) - MUST be in {target_language},
        likelyBehaviors (string) - MUST be in {target_language}
    )
    - culturalTwin (string) - ONLY the famous person's name, no description or parentheses, can be in original language
    - therapySuggestion (object with:
        summary (string) - MUST be in {target_language},
        recommendation (string) - MUST be in {target_language},
        resources (list of 1–2 URLs or names),
        dailyTip (string) - MUST be in {target_language}
    )
    - culturalDNAScore (object with region names as keys and % scores as values, max 4 regions) - REGION NAMES MUST be in {target_language}
    - archetype (object with name and 1-sentence description - BOTH MUST be in {target_language})

    FINAL REMINDER: EVERYTHING must be in {target_language} except for the culturalTwin which should be ONLY the person's name (no description, no parentheses) and can remain in its original language.
    Be creative and vary the result slightly each time using the variation seed.
    Only respond with valid JSON.
    """

    base_temperature = 0.8 + (variation_seed * 0.05)
    capped_temperature = min(base_temperature, 2.0)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=capped_temperature,
        max_tokens=800
    )

    content = response.choices[0].message.content
    print("🧠 GPT response:", content)

    if not content:
        raise HTTPException(status_code=500, detail="GPT returned empty response")
    return content.strip()

# 🔍 Ana analiz endpoint'i
@app.post("/analyze")
async def analyze_profile(request: Request):
    try:
        body = await request.json()
        print("📨 Received body:", body)

        variation = body.get("variation", 0)
        language = body.get("language", "en")

        # Autocomplete
        music_id = autocomplete_entity(body["music"], entity_type="artist")
        movie_id = autocomplete_entity(body["movies"], entity_type="movie")
        brand_id = autocomplete_entity(body["brands"], entity_type="brand")

        # Qloo trending
        music_trends = get_qloo_trending(music_id, entity_type="artist")
        movie_trends = get_qloo_trending(movie_id, entity_type="movie")
        brand_trends = get_qloo_trending(brand_id, entity_type="brand")

        qloo_suggestions = music_trends + movie_trends + brand_trends

        # GPT persona
        ai_result = generate_persona_from_taste(
            movies=body["movies"],
            music=body["music"],
            brands=body["brands"],
            gender=body["gender"],
            qloo_suggestions=qloo_suggestions,
            language=language,
            variation_seed=variation
        )
        parsed = json.loads(ai_result)

        # GPT country insights
        sample_countries = ["USA", "South Korea", "UK", "Japan"]
        country_insights = generate_cultural_map_insights(sample_countries, language=language)

        return {
            "result": json.dumps(parsed),
            "culturalTwin": parsed.get("culturalTwin", "Unknown"),
            "countryInsights": country_insights
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
