"""
Axis AI Bot — Vercel Backend v2.0
Vision-based iterative planning.

Extension → screenshot bhejta hai → Vision AI (Nemotron Omni) scene dekhta hai
→ Objects ka real position samajhta hai → Precise action plan return karta hai.
"""

import os, json, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Axis AI Bot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────
class CanvasRect(BaseModel):
    x: float; y: float; w: float; h: float

class PlanRequest(BaseModel):
    task_name:          str
    steps:              list[str]
    difficulty:         Optional[str]   = None
    canvas:             Optional[CanvasRect] = None
    url:                Optional[str]   = None
    openrouter_key:     Optional[str]   = None
    screenshot_b64:     Optional[str]   = None   # Canvas screenshot (base64 JPEG)
    previous_analysis:  Optional[str]   = None   # Pichli iteration ka context
    iteration:          Optional[int]   = 1

# ─────────────────────────────────────────────────
# OPENROUTER CONFIG
# ─────────────────────────────────────────────────
OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL    = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"  # Vision + reasoning
FALLBACK_MODEL  = "meta-llama/llama-4-scout:free"                        # Text fallback

# ─────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot arm controller AI for Axis Robotics simulation.
You will receive a screenshot of the robot simulation canvas and must analyze it to complete the given task.

YOUR JOB:
1. Look at the screenshot carefully
2. Identify the robot arm's current position (blue control ring)
3. Identify all task objects (cups, bottles, plates, etc.) and their canvas positions
4. Identify the target area if visible
5. Plan precise actions based on ACTUAL positions you see

CANVAS COORDINATE SYSTEM:
- x, y are 0.0 to 1.0 (relative to canvas width/height)
- Top-left = (0,0), Bottom-right = (1,1)
- Objects are typically in lower half (y > 0.5)
- The blue ring on the arm is the grab control point

AVAILABLE ACTIONS:
{"type":"double_click","x":0.4,"y":0.6,"wait_ms":1500,"reason":"..."}
  → Pre-grasp pose toward that position (double-click blue ring / object)

{"type":"key_hold","key":"ArrowRight","duration_ms":600,"wait_ms":200,"reason":"..."}
  → Move arm. Keys: ArrowUp(forward) ArrowDown(back) ArrowLeft(left) ArrowRight(right) e(up) d(down)

{"type":"gripper","wait_ms":800,"reason":"open or close gripper"}
  → Toggle gripper (SPACE). Do this AFTER arm is near object.

{"type":"switch_arm","wait_ms":400,"reason":"..."}
  → Switch to other arm (C key) — for bimanual tasks

{"type":"drag","from_x":0.4,"from_y":0.6,"to_x":0.6,"to_y":0.4,"wait_ms":600,"reason":"..."}
  → Drag from one point to another on canvas

{"type":"checkpoint_save","reason":"..."}  → Save good state (N)
{"type":"checkpoint_restore","reason":"..."} → Restore if stuck (B)
{"type":"wait","ms":1500}  → Wait for physics

STRATEGY FOR PICK AND PLACE:
1. double_click on the object you want to pick (use its actual x,y from screenshot)
2. wait 1500ms for arm to move there
3. gripper (close)
4. key_hold "e" 600ms (lift up)
5. Move arm to target position using key_hold (ArrowLeft/Right/Up/Down)
6. key_hold "d" 400ms (lower down)
7. gripper (open) — object placed
8. key_hold "e" 400ms (lift arm away)

IMPORTANT:
- Only plan 4-6 actions per iteration (you will see new screenshot after)
- Use real coordinates from the screenshot, not guesses
- If previous iteration failed, try different approach
- Return is_complete: true only when ALL task objects are correctly placed

RESPOND ONLY WITH THIS JSON (no markdown, no explanation):
{
  "analysis": "What you see in the screenshot: arm position, object positions, what needs to be done",
  "actions": [...],
  "is_complete": false
}"""

# ─────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "version": "2.0", "model": VISION_MODEL}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/plan")
async def get_plan(req: PlanRequest):
    api_key = req.openrouter_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(400, "OpenRouter API key missing")

    # User message banao — text + optional image
    user_content = build_user_content(req)

    # OpenRouter call
    result, model_used = await call_openrouter(api_key, user_content, has_image=bool(req.screenshot_b64))

    return {**result, "model": model_used}


# ─────────────────────────────────────────────────
# MESSAGE BUILDER
# ─────────────────────────────────────────────────
def build_user_content(req: PlanRequest):
    """Vision model ke liye multimodal content banao."""
    text_parts = []

    text_parts.append(f"TASK: {req.task_name}")
    if req.difficulty:
        text_parts.append(f"DIFFICULTY: {req.difficulty}")

    if req.steps:
        text_parts.append("STEPS TO COMPLETE:")
        for i, s in enumerate(req.steps, 1):
            text_parts.append(f"  {i}. {s}")

    if req.canvas:
        text_parts.append(f"CANVAS SIZE: {req.canvas.w:.0f}x{req.canvas.h:.0f}px")

    text_parts.append(f"ITERATION: {req.iteration} (plan only 4-6 actions, you'll get new screenshot after)")

    if req.previous_analysis:
        text_parts.append(f"\nPREVIOUS ITERATION RESULT:\n{req.previous_analysis}")
        text_parts.append("→ Continue from where you left off. Adjust if needed.")

    text_parts.append("\nAnalyze the screenshot and return the JSON plan.")

    text = "\n".join(text_parts)

    # Screenshot hai → multimodal content
    if req.screenshot_b64:
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{req.screenshot_b64}"
                }
            },
            {
                "type": "text",
                "text": text
            }
        ]
    else:
        # No screenshot → text only
        return text


# ─────────────────────────────────────────────────
# OPENROUTER CALL
# ─────────────────────────────────────────────────
async def call_openrouter(api_key: str, user_content, has_image: bool):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://axis-ai-bot.vercel.app",
        "X-Title": "Axis AI Bot",
    }

    model = VISION_MODEL if has_image else FALLBACK_MODEL

    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)

            if resp.status_code == 200:
                return parse_response(resp.json()), model

            print(f"[AxisAI] {model} error {resp.status_code}. Trying fallback...")

        except httpx.TimeoutException:
            print(f"[AxisAI] {model} timeout. Trying fallback...")

        # Fallback — text only
        payload["model"] = FALLBACK_MODEL
        # Image hata do fallback ke liye
        if has_image and isinstance(user_content, list):
            text_only = next((c["text"] for c in user_content if c["type"] == "text"), "")
            payload["messages"][-1]["content"] = text_only + "\n(Note: No screenshot available, use standard strategy)"

        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, f"OpenRouter error: {resp.status_code} — {resp.text[:300]}")

        return parse_response(resp.json()), FALLBACK_MODEL


# ─────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────
def parse_response(response: dict) -> dict:
    try:
        content = response["choices"][0]["message"]["content"].strip()

        # Markdown backticks hato
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Reasoning tags hato (Nemotron reasoning model ke liye)
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()

        data = json.loads(content)

        # Validate
        valid_types = {
            "double_click","click","drag","key","key_hold",
            "gripper","switch_arm","checkpoint_save","checkpoint_restore","wait"
        }
        actions = [
            a for a in data.get("actions", [])
            if isinstance(a, dict) and a.get("type") in valid_types
        ]

        return {
            "analysis":    data.get("analysis", ""),
            "actions":     actions if actions else default_actions(),
            "is_complete": data.get("is_complete", False),
        }

    except Exception as e:
        print(f"[AxisAI] Parse error: {e}. Using default.")
        return {
            "analysis": "Parse error — using fallback strategy",
            "actions":  default_actions(),
            "is_complete": False,
        }


def default_actions() -> list:
    """Fallback jab AI parse fail ho."""
    return [
        {"type": "wait",         "ms": 1000},
        {"type": "double_click", "x": 0.38, "y": 0.68, "wait_ms": 2000, "reason": "grab object 1"},
        {"type": "gripper",      "wait_ms": 800, "reason": "close gripper"},
        {"type": "key_hold",     "key": "e", "duration_ms": 700, "reason": "lift up"},
        {"type": "key_hold",     "key": "ArrowRight", "duration_ms": 600, "reason": "move right"},
        {"type": "key_hold",     "key": "d", "duration_ms": 400, "reason": "lower down"},
        {"type": "gripper",      "wait_ms": 800, "reason": "release"},
        {"type": "key_hold",     "key": "e", "duration_ms": 500, "reason": "lift back"},
    ]
