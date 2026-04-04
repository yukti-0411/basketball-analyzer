import json
import os
import urllib.error
from typing import Optional
import requests

IDEAL_ELBOW_MIN_DEG = 150.0
IDEAL_KNEE_MIN_DEG = 160.0

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _build_rule_observations(
    elbow_angle_deg, knee_angle_deg, wrist_above_shoulder, elbow_offset_px, flare_threshold_px
):
    lines = []

    if elbow_angle_deg is None:
        lines.append("Elbow angle: not measured.")
    elif elbow_angle_deg >= IDEAL_ELBOW_MIN_DEG:
        lines.append(f"Elbow angle ({elbow_angle_deg:.1f}°): GOOD — arm fairly straight at release.")
    else:
        lines.append(f"Elbow angle ({elbow_angle_deg:.1f}°): NEEDS WORK — aim for {IDEAL_ELBOW_MIN_DEG:.0f}°+.")

    if knee_angle_deg is None:
        lines.append("Knee angle: not measured.")
    elif knee_angle_deg >= IDEAL_KNEE_MIN_DEG:
        lines.append(f"Knee angle ({knee_angle_deg:.1f}°): GOOD — legs extended from jump.")
    else:
        lines.append(f"Knee angle ({knee_angle_deg:.1f}°): NEEDS WORK — extend more through the legs.")

    if wrist_above_shoulder is None:
        lines.append("Wrist vs shoulder: not measured.")
    elif wrist_above_shoulder:
        lines.append("Wrist above shoulder: GOOD — release height looks correct.")
    else:
        lines.append("Wrist above shoulder: NEEDS WORK — wrist should be above shoulder at release.")

    if elbow_offset_px is None:
        lines.append("Elbow alignment: not measured.")
    else:
        abs_off = abs(elbow_offset_px)
        if abs_off <= flare_threshold_px:
            lines.append("Elbow alignment: GOOD — elbow not flared out.")
        else:
            lines.append("Elbow alignment: NEEDS WORK — elbow is flared, keep it under the ball.")

    return lines


def _call_groq_api(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a supportive basketball coach. Be concise, encouraging, and specific. "
                    "Use the observations exactly. End with one short drill or cue."
                )
            },
            {"role": "user", "content": prompt}
        ],
    }
    response = requests.post(GROQ_URL, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_feedback(
    elbow_angle_deg, knee_angle_deg, wrist_above_shoulder, elbow_offset_px, flare_threshold_px, api_key=None
):
    observations = _build_rule_observations(
        elbow_angle_deg, knee_angle_deg, wrist_above_shoulder, elbow_offset_px, flare_threshold_px
    )

    print("\n--- Rule-based observations (release frame) ---")
    for line in observations:
        print(line)

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        print("\n(No GROQ_API_KEY — skipping coaching message.)")
        return None

    obs_text = "\n".join(f"{i+1}. {o}" for i, o in enumerate(observations))
    prompt = (
        "Here are observations from a basketball release frame analysis. "
        "Write a short personalized coaching paragraph.\n\n"
        f"{obs_text}\n\n"
        "Acknowledge what looks good first, then what to improve, in friendly plain language."
    )

    try:
        coaching = _call_groq_api(prompt, key)
    except requests.exceptions.HTTPError as e:
        print(f"\nGroq API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        return None

    print("\n--- Coaching message ---")
    print(coaching)
    return coaching