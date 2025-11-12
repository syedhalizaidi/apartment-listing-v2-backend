from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

client = OpenAI()

class SearchPrefs(BaseModel):
    bedrooms: Optional[List[int]] = Field(default=None)
    max_budget_usd: Optional[int] = None
    move_date: Optional[str] = None
    areas: Optional[List[str]] = None
    dog_friendly: Optional[bool] = None
    amenities: Optional[List[str]] = None
    vibe: Optional[str] = None
    selected_indices: Optional[List[int]] = None
    schedule_days: Optional[List[str]] = None
    time_pref: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None


INSTRUCTIONS = """Extract user home search preferences from the message.
Return only fields present; omit unknowns.
- bedrooms: list of ints like [2,3]
- max_budget_usd: integer for max monthly rent
- move_date: ISO or natural language date
- areas: list of neighborhoods/areas/cities mentioned
- dog_friendly: true/false if user says dog-friendly required; else null
- amenities: list like ["high ceilings","wood floors","balcony","large kitchen"]
- vibe: short words like "residential", "nightlife", "quiet"
- selected_indices: from phrases like "first and third options" -> [1,3]
- schedule_days: ["Tuesday","Wednesday"]
- time_pref: "afternoons", etc.
- contact_email, contact_phone: extract if present"""

def parse_prefs(message: str) -> SearchPrefs:
    resp = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role":"system","content":INSTRUCTIONS},
            {"role":"user","content":message}
        ],
        text_format=SearchPrefs
    )
    return resp.output_parsed


# -----------------------------
# Simple, reusable GPT wrapper
# -----------------------------

def get_gpt_response(client: OpenAI, prompt: str, system_commands: Dict[str, Any]) -> Any:
    """Optimized GPT Response Retrieval.

    system_commands shape:
    {
      "role": str,
      "tone": str,
      "knowledge": str,
      "do": str,
      "dont": str
    }
    """
    role = (system_commands or {}).get("role", "assistant")
    tone = (system_commands or {}).get("tone", "Professional")
    knowledge = (system_commands or {}).get("knowledge", "")
    do = (system_commands or {}).get("do", "")
    dont = (system_commands or {}).get("dont", "")

    sys = (
        f"You are an assistant. "
        f"Role: {role} "
        f"Tone: {tone} "
        f"Knowledge: {knowledge} "
        f"Do: {do} "
        f"Don't: {dont}"
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000
        )
        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        logging.error(f"Error while generating GPT response: {e}")
        return {"error": "An error occurred while processing the response."}
