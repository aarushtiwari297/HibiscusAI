"""
Hibiscus Leaf Visual Symptom Analysis - Gemini API Integration Module
======================================================================
Provides second-stage visual symptom category classification using the
official Google Gen AI SDK (`google-genai`).

Supported symptom categories:
  - Healthy
  - Chewed
  - Dried
  - Leaf Spot
  - Yellowing
  - Other / Uncertain

Note: These categories describe visible symptoms or physical damage patterns
and are strictly distinct from confirmed biological disease diagnoses.
"""

import os
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Explicitly load backend/.env with override=True to prevent stale or empty parent env vars from overriding it
ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_FILE_PATH, override=True)

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SUPPORTED_CATEGORIES = [
    "Healthy",
    "Chewed",
    "Dried",
    "Leaf Spot",
    "Yellowing",
    "Other / Uncertain"
]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

KNOWN_PLACEHOLDERS = {
    "your_actual_gemini_api_key_here",
    "your_gemini_api_key_here",
    "your_api_key_here",
    "<your_api_key>",
    "<your_gemini_api_key>",
    "changeme",
    "placeholder"
}


class SymptomCategory(str, Enum):
    HEALTHY = "Healthy"
    CHEWED = "Chewed"
    DRIED = "Dried"
    LEAF_SPOT = "Leaf Spot"
    YELLOWING = "Yellowing"
    OTHER_UNCERTAIN = "Other / Uncertain"


class HibiscusSymptomAnalysis(BaseModel):
    category: SymptomCategory = Field(
        description="Visible symptom category from the supported list."
    )
    visible_symptoms: List[str] = Field(
        description="Specific visible symptoms, damage marks, or image quality observations."
    )
    explanation: str = Field(
        description="Short explanation (1-3 sentences) detailing the visible evidence."
    )
    uncertain: bool = Field(
        description="True if blur, lighting, non-leaf subject, or ambiguous patterns prevent reliable classification."
    )
    confidence: str = Field(
        description="Assessment confidence level: 'high', 'medium', or 'low'."
    )


SYSTEM_INSTRUCTION = """You are an expert botanical image analyst specializing in hibiscus (Hibiscus rosa-sinensis) foliage.
Analyze the provided image of a hibiscus leaf or plant to identify visible visual symptoms or damage patterns.

CRITICAL RULES:
1. Do NOT diagnose a biological disease or name specific pathogens as absolute facts; focus strictly on visible symptom patterns.
2. The category MUST strictly be one of:
   - "Healthy": Vibrant, intact, green leaf with no visible lesions, insect damage, or drying.
   - "Chewed": Visible mechanical insect/caterpillar bite marks, holes, serrated missing tissue.
   - "Dried": Desiccated, curled, brittle, scorched, or necrotic leaf tips/edges.
   - "Leaf Spot": Discrete localized circular or irregular spots, fungal or bacterial lesions, dark necrotic specks.
   - "Yellowing": Chlorosis, overall or patchy loss of chlorophyll, pale/yellow pigmentation.
   - "Other / Uncertain": Image does not show a plant/hibiscus leaf, is too blurry, too dark, out of focus, or has ambiguous mixed patterns.
3. If the image is blurry, poorly lit, low resolution, or does NOT depict a plant/leaf, YOU MUST assign category="Other / Uncertain", set uncertain=true, and describe the limitation in visible_symptoms and explanation.
4. Keep the explanation concise (2-3 sentences max) based solely on visual evidence in the image.
"""


def is_gemini_configured() -> bool:
    """Checks whether GEMINI_API_KEY is configured with a valid, non-empty, non-placeholder value."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key.lower() in KNOWN_PLACEHOLDERS:
        return False
    return True


def get_gemini_diagnostics() -> Dict[str, Any]:
    """
    Returns safe diagnostic status without revealing the secret key, prefix, or length.
    """
    env_exists = os.path.isfile(ENV_FILE_PATH)
    raw_key = os.environ.get("GEMINI_API_KEY", "").strip()
    is_non_empty = bool(raw_key)
    is_placeholder = raw_key.lower() in KNOWN_PLACEHOLDERS
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    return {
        "env_file_exists": env_exists,
        "key_configured": is_non_empty and not is_placeholder,
        "is_placeholder": is_placeholder,
        "is_non_empty": is_non_empty,
        "selected_model": model_name
    }



def analyze_leaf_symptoms(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends leaf image to Gemini vision model to classify visible symptom categories.

    Returns structured dict with:
      - success: bool
      - category: str (strictly one of SUPPORTED_CATEGORIES)
      - visible_symptoms: list[str]
      - explanation: str
      - uncertain: bool
      - confidence: str
      - model: str
      - disclaimer: str
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "success": False,
            "error": "Gemini API key is not configured. Set GEMINI_API_KEY in the backend environment.",
            "code": "MISSING_API_KEY"
        }

    selected_model = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    try:
        client = genai.Client(api_key=api_key)

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        user_prompt = (
            "Analyze this image and identify the visible hibiscus foliage symptom category. "
            "Return structured JSON matching the defined schema."
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=HibiscusSymptomAnalysis,
            temperature=0.2,
        )

        # Resilient model fallback: if primary model is experiencing high demand (503), try gemini-1.5-flash
        models_to_try = [selected_model]
        if selected_model != "gemini-1.5-flash":
            models_to_try.append("gemini-1.5-flash")

        response = None
        used_model = selected_model

        for current_model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=[image_part, user_prompt],
                    config=config,
                )
                used_model = current_model
                break
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str) and current_model != models_to_try[-1]:
                    logger.warning("Model %s temporarily unavailable (503). Retrying with fallback %s...", current_model, models_to_try[-1])
                    continue
                raise

        if not response or not response.text:

            return {
                "success": False,
                "error": "Gemini returned an empty response.",
                "code": "EMPTY_RESPONSE"
            }

        parsed_data = json.loads(response.text)

        # Validate category against supported list
        raw_category = parsed_data.get("category", "")
        if raw_category not in SUPPORTED_CATEGORIES:
            logger.warning("Gemini returned unsupported category: '%s'. Coercing to 'Other / Uncertain'.", raw_category)
            category = "Other / Uncertain"
            uncertain = True
        else:
            category = raw_category
            uncertain = bool(parsed_data.get("uncertain", False))

        visible_symptoms = parsed_data.get("visible_symptoms") or []
        if isinstance(visible_symptoms, str):
            visible_symptoms = [visible_symptoms]

        explanation = parsed_data.get("explanation") or ""
        confidence = parsed_data.get("confidence") or ("low" if uncertain else "medium")

        return {
            "success": True,
            "category": category,
            "visible_symptoms": visible_symptoms,
            "explanation": explanation,
            "uncertain": uncertain,
            "confidence": confidence,
            "model": used_model,
            "disclaimer": (
                "Gemini-assisted visual symptom classification screens observable damage patterns only "
                "and does not constitute a certified biological laboratory diagnosis."
            )
        }

    except Exception as exc:
        err_str = str(exc)
        logger.error("Gemini symptom analysis failed: %s: %s", type(exc).__name__, err_str)

        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return {
                "success": False,
                "error": "Gemini API rate limit or quota exceeded. Please try again shortly.",
                "code": "QUOTA_EXCEEDED"
            }
        elif "401" in err_str or "403" in err_str or "API_KEY_INVALID" in err_str:
            return {
                "success": False,
                "error": "Invalid or unauthorized Gemini API key.",
                "code": "INVALID_API_KEY"
            }
        elif "timeout" in err_str.lower() or "deadline" in err_str.lower():
            return {
                "success": False,
                "error": "Gemini API request timed out. Please try again.",
                "code": "TIMEOUT"
            }
        else:
            return {
                "success": False,
                "error": f"Gemini visual analysis encountered an error: {type(exc).__name__}",
                "code": "GEMINI_API_ERROR"
            }
