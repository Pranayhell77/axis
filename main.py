"""
Axis AI Bot — Vercel Backend
FastAPI + OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free)

Extension se task info aata hai → Nemotron se action plan banata hai → JSON return karta hai.
"""

import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Axis AI Bot Backend", version="1.0.0")

# ─────────────────────────────────────────────────
# CORS — Browser extension se calls allow karo
# ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Extension kisi bhi origin se call kar sakti hai
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────
class CanvasRect(BaseModel):
    x: float
    y: float
    w: float
    h: float

class TaskRequest(BaseModel):
    task_name:      str
    steps:          list[str]
    difficulty:     Optional[str] = None
    canvas:         Optional[CanvasRect] = None
    url:            Optional[str] = None
    openrouter_key: Optional[str] = None   # Extension se aata hai (fallback)

class ActionPlanResponse(BaseModel):
    actions: list[dict]
    model:   str
    task:    str

# ─────────────────────────────────────────────────
# OPENROUTER CONFIG
# ─────────────────────────────────────────────────
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL    = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_MODEL   = "meta-llama/llama-4-scout:free"   # Nemotron rate limit ho to

# ─────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot arm controller AI for Axis Robotics simulation.
Your job: read a task description and return a JSON array of actions to complete it.

AVAILABLE ACTION TYPES:
- {"type":"double_click","x":0.4,"y":0.6,"wait_ms":1500,"reason":"..."}
  → Double-click at canvas position (x,y are 0.0–1.0 relative to canvas)
- {"type":"drag","from_x":0.3,"from_y":0.6,"to_x":0.6,"to_y":0.4,"steps":25,"wait_ms":600,"reason":"..."}
  → Smooth drag from one point to another
- {"type":"key_hold","key":"ArrowUp","duration_ms":500,"wait_ms":200,"reason":"..."}
  → Hold a key for duration (for moving arm)
  Keys: ArrowUp ArrowDown ArrowLeft ArrowRight e d q w (for z-axis)
- {"type":"gripper","wait_ms":600,"reason":"..."}
  → Toggle gripper open/close (SPACE key)
- {"type":"switch_arm","wait_ms":400,"reason":"..."}
  → Switch between arms for bimanual tasks (C key)
- {"type":"checkpoint_save","reason":"..."}
  → Save checkpoint (N key) — use after a good position
- {"type":"checkpoint_restore","reason":"..."}
  → Restore checkpoint (B key) — use if stuck
- {"type":"wait","ms":1000}
  → Wait for animation/physics to settle

ROBOT CONTROLS REFERENCE:
- Arrow keys: move arm left/right/forward/back in XY plane
- E/D keys: move arm up/down (Z axis)
- Q/W keys: rotate arm
- SPACE: open/close gripper
- C: switch arm (bimanual)
- Double-click on canvas: pre-grasp pose toward that object

CANVAS POSITIONS (approximate):
- Objects are usually in lower-center area: x≈0.3–0.5, y≈0.5–0.7
- Target zones are usually upper or right area: x≈0.5–0.7, y≈0.3–0.5
- Center of canvas: x=0.5, y=0.5

RULES:
1. Always open gripper before approaching object
2. Double-click object → wait → close gripper → lift (E key) → move to target → lower (D key) → open gripper
3. Use checkpoint_save after grabbing object successfully
4. Add wait_ms after each physical action (1000-2000ms for physics to settle)
5. For bimanual tasks, use switch_arm between left/right arm actions

RESPOND ONLY WITH A VALID JSON ARRAY. No explanation, no markdown, no text before or after."""

# ─────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "Axis AI Bot Backend", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/plan", response_model=ActionPlanResponse)
async def get_action_plan(req: TaskRequest):
    """
    Extension se task info receive karo,
    Nemotron se action plan lo,
    JSON actions return karo.
    """

    # API key — extension se ya environment se
    api_key = req.openrouter_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key missing. Extension popup mein set karo ya OPENROUTER_API_KEY env var use karo."
        )

    # User message banao
    user_message = build_user_message(req)

    # OpenRouter call karo — pehle Nemotron, fail ho to Llama fallback
    actions, model_used = await call_openrouter(api_key, user_message)

    return ActionPlanResponse(
        actions=actions,
        model=model_used,
        task=req.task_name,
    )


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────
def build_user_message(req: TaskRequest) -> str:
    """Task info se concise prompt banao."""
    lines = [f"TASK: {req.task_name}"]

    if req.difficulty:
        lines.append(f"DIFFICULTY: {req.difficulty}")

    if req.steps:
        lines.append("STEPS TO COMPLETE:")
        for i, step in enumerate(req.steps, 1):
            lines.append(f"  {i}. {step}")
    else:
        lines.append("STEPS: Not specified — use standard pick-and-place strategy")

    if req.canvas:
        lines.append(f"CANVAS SIZE: {req.canvas.w:.0f}x{req.canvas.h:.0f}px")

    lines.append("\nGenerate the JSON action array to complete this task.")
    return "\n".join(lines)


async def call_openrouter(api_key: str, user_message: str) -> tuple[list, str]:
    """
    OpenRouter API call karo.
    Pehle Nemotron try karo, rate limit ya error aaye to Llama fallback.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://axis-ai-bot.vercel.app",
        "X-Title": "Axis AI Bot",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 800,
        "temperature": 0.2,      # Low temp = consistent JSON output
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        # ── Try 1: Nemotron ──
        try:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)

            if resp.status_code == 200:
                return parse_llm_response(resp.json()), DEFAULT_MODEL

            # Rate limited ya model unavailable → fallback
            if resp.status_code in (429, 503):
                print(f"[AxisAI] {DEFAULT_MODEL} rate limited ({resp.status_code}). Fallback to Llama...")
            else:
                print(f"[AxisAI] Nemotron error {resp.status_code}: {resp.text[:200]}")

        except httpx.TimeoutException:
            print("[AxisAI] Nemotron timeout. Fallback to Llama...")

        # ── Try 2: Llama Fallback ──
        payload["model"] = FALLBACK_MODEL
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter error: {resp.status_code} — {resp.text[:300]}"
            )

        return parse_llm_response(resp.json()), FALLBACK_MODEL


def parse_llm_response(response: dict) -> list:
    """LLM response se JSON action array extract karo."""
    try:
        content = response["choices"][0]["message"]["content"].strip()

        # Markdown backticks hato
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        actions = json.loads(content)

        # Validate — list of dicts honi chahiye
        if not isinstance(actions, list):
            raise ValueError("Response list nahi hai")

        # Sirf valid action types rakhao
        valid_types = {
            "double_click", "click", "drag", "key", "key_hold",
            "gripper", "switch_arm", "checkpoint_save",
            "checkpoint_restore", "wait"
        }
        actions = [a for a in actions if isinstance(a, dict) and a.get("type") in valid_types]

        if not actions:
            raise ValueError("Koi valid actions nahi mile")

        return actions

    except (KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"[AxisAI] Parse error: {e}. Default strategy use ho rahi hai.")
        return default_pick_and_place()


def default_pick_and_place() -> list:
    """LLM fail ho to yeh fallback actions use karo."""
    return [
        {"type": "wait",         "ms": 1500},
        {"type": "double_click", "x": 0.35, "y": 0.65, "wait_ms": 2000, "reason": "object pre-grasp"},
        {"type": "gripper",      "wait_ms": 800,                         "reason": "close gripper"},
        {"type": "key_hold",     "key": "e", "duration_ms": 800,         "reason": "lift up"},
        {"type": "wait",         "ms": 500},
        {"type": "double_click", "x": 0.65, "y": 0.40, "wait_ms": 2000, "reason": "move to target"},
        {"type": "key_hold",     "key": "d", "duration_ms": 400,         "reason": "lower down"},
        {"type": "gripper",      "wait_ms": 800,                         "reason": "open gripper"},
        {"type": "key_hold",     "key": "e", "duration_ms": 600,         "reason": "lift arm back"},
        {"type": "wait",         "ms": 1000},
    ]
