"""
Axis AI Bot Backend v3.0
3 endpoints: /api/scan  /api/plan  /api/verify
"""
import os, json, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Axis AI Bot", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL   = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
TEXT_MODEL     = "meta-llama/llama-4-scout:free"

# ── Pydantic models ──
class CanvasRect(BaseModel):
    x:float; y:float; w:float; h:float

class BaseReq(BaseModel):
    task_name:      str
    steps:          list[str]
    canvas:         Optional[CanvasRect] = None
    openrouter_key: Optional[str]        = None
    screenshot_b64: Optional[str]        = None

class ScanReq(BaseReq): pass

class PlanReq(BaseReq):
    previous_analysis: Optional[str] = None
    iteration:         Optional[int] = 1

class VerifyReq(BaseReq):
    previous_scan: Optional[str] = None

# ── Routes ──
@app.get("/")
async def root(): return {"status":"ok","version":"3.0"}

@app.get("/health")
async def health(): return {"status":"healthy"}


# ════════════════════════════════════════════════
# /api/scan — Phase 1: Positions scan karo
# ════════════════════════════════════════════════
@app.post("/api/scan")
async def scan(req: ScanReq):
    """
    Screenshot dekh ke sab positions return karo:
    arm, objects (name+x+y), target area
    """
    key = req.openrouter_key or os.getenv("OPENROUTER_API_KEY","")
    if not key: raise HTTPException(400,"API key missing")

    system = """You are a robot vision AI. Analyze the simulation screenshot and identify positions.

CANVAS COORDINATE SYSTEM:
- x, y = 0.0 to 1.0 (relative to canvas: left=0, right=1, top=0, bottom=1)
- The robot arm is the mechanical arm structure
- Objects to manipulate are highlighted (they may have a blue/yellow glow or label)
- Target area is usually marked with yellow dots or a highlighted zone

Return ONLY this JSON (no markdown, no explanation):
{
  "arm": {"x": 0.5, "y": 0.6, "description": "arm ki current position"},
  "objects": [
    {"name": "coffee cup", "x": 0.38, "y": 0.72, "description": "brown cup near center"},
    {"name": "water bottle", "x": 0.52, "y": 0.68, "description": "tall blue bottle"},
    {"name": "plastic bottle", "x": 0.45, "y": 0.75, "description": "small bottle right side"}
  ],
  "target": {"x": 0.5, "y": 0.6, "description": "target zone description"},
  "notes": "Task ko complete karne ke liye kya karna hai — brief"
}

IMPORTANT: x,y positions must be accurate based on what you see in the image.
If no screenshot, estimate based on task description."""

    user_content = build_content(req, f"""Task: {req.task_name}
Steps: {json.dumps(req.steps)}

Analyze the screenshot and return positions of:
1. Robot arm (blue control ring)
2. All task objects (cups, bottles, etc.)
3. Target area where objects need to go

Return JSON with positions.""")

    raw, model = await call_or(key, system, user_content, bool(req.screenshot_b64))

    try:
        data = parse_json(raw)
        return {
            "arm":     data.get("arm", {}),
            "objects": data.get("objects", []),
            "target":  data.get("target", {}),
            "notes":   data.get("notes", ""),
            "model":   model,
        }
    except Exception as e:
        print(f"[Scan parse error] {e}")
        return {"arm":{},"objects":[],"target":{"x":0.5,"y":0.55},"notes":"Parse error","model":model}


# ════════════════════════════════════════════════
# /api/plan — Phase 2 fallback: action plan lo
# ════════════════════════════════════════════════
@app.post("/api/plan")
async def plan(req: PlanReq):
    key = req.openrouter_key or os.getenv("OPENROUTER_API_KEY","")
    if not key: raise HTTPException(400,"API key missing")

    system = """You are a robot arm controller. Look at the screenshot and plan actions.

DOUBLE-CLICK STRATEGY (preferred):
- double_click on an object → arm automatically moves there
- Then close gripper → double_click target → open gripper

ACTIONS:
{"type":"double_click","x":0.4,"y":0.6,"wait_ms":2500,"reason":"..."}
{"type":"gripper","wait_ms":800,"reason":"open/close"}
{"type":"key_hold","key":"ArrowRight","duration_ms":500,"wait_ms":200,"reason":"..."}
  Keys: ArrowUp ArrowDown ArrowLeft ArrowRight e(up) d(down)
{"type":"switch_arm","wait_ms":400,"reason":"..."}
{"type":"checkpoint_save","reason":"..."}
{"type":"wait","ms":1000}

Return ONLY JSON:
{
  "analysis": "kya dekha, kya plan hai",
  "actions": [...],
  "is_complete": false
}"""

    prev = f"\nPrevious: {req.previous_analysis}" if req.previous_analysis else ""
    user_content = build_content(req, f"""Task: {req.task_name}
Steps: {json.dumps(req.steps)}
Iteration: {req.iteration}{prev}

Plan 4-6 actions using double-click strategy. Return JSON.""")

    raw, model = await call_or(key, system, user_content, bool(req.screenshot_b64))

    try:
        data = parse_json(raw)
        actions = [a for a in data.get("actions",[]) if isinstance(a,dict) and a.get("type") in VALID_ACTIONS]
        return {
            "analysis": data.get("analysis",""),
            "actions":  actions or default_actions(),
            "is_complete": data.get("is_complete", False),
            "model": model,
        }
    except Exception as e:
        print(f"[Plan parse error] {e}")
        return {"analysis":"Parse error","actions":default_actions(),"is_complete":False,"model":model}


# ════════════════════════════════════════════════
# /api/verify — Phase 3: check + adjustment
# ════════════════════════════════════════════════
@app.post("/api/verify")
async def verify(req: VerifyReq):
    key = req.openrouter_key or os.getenv("OPENROUTER_API_KEY","")
    if not key: raise HTTPException(400,"API key missing")

    system = """You are a robot task verifier. Check if the task is complete.
Look at the current screenshot and compare with the task goal.

Return ONLY JSON:
{
  "is_complete": false,
  "notes": "kya hua, kya baki hai",
  "needs_adjustment": true,
  "actions": [
    {"type":"double_click","x":0.4,"y":0.6,"wait_ms":2000,"reason":"object theek karo"}
  ]
}

If task is complete, set is_complete: true and actions: []
If adjustment needed, provide 2-3 corrective actions max."""

    prev = f"\nPrevious scan: {req.previous_scan}" if req.previous_scan else ""
    user_content = build_content(req, f"""Task: {req.task_name}
Steps: {json.dumps(req.steps)}{prev}

Is the task complete? If not, what small adjustments are needed?""")

    raw, model = await call_or(key, system, user_content, bool(req.screenshot_b64))

    try:
        data = parse_json(raw)
        actions = [a for a in data.get("actions",[]) if isinstance(a,dict) and a.get("type") in VALID_ACTIONS]
        return {
            "is_complete":       data.get("is_complete", False),
            "notes":             data.get("notes", ""),
            "needs_adjustment":  data.get("needs_adjustment", False),
            "actions":           actions,
            "model":             model,
        }
    except Exception as e:
        print(f"[Verify parse error] {e}")
        return {"is_complete":False,"notes":"Parse error","needs_adjustment":False,"actions":[],"model":model}


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────
VALID_ACTIONS = {"double_click","click","drag","key","key_hold","gripper","switch_arm","checkpoint_save","checkpoint_restore","wait"}

def build_content(req: BaseReq, text: str):
    if req.screenshot_b64:
        return [
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{req.screenshot_b64}"}},
            {"type":"text","text":text}
        ]
    return text

async def call_or(key:str, system:str, user_content, has_image:bool):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://axis-ai-bot.vercel.app",
        "X-Title": "Axis AI Bot",
    }
    model = VISION_MODEL if has_image else TEXT_MODEL
    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.1,
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":user_content},
        ]
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], model
            print(f"[{model}] {r.status_code} — fallback")
        except httpx.TimeoutException:
            print(f"[{model}] timeout — fallback")

        payload["model"] = TEXT_MODEL
        if has_image and isinstance(user_content, list):
            text_only = next((c["text"] for c in user_content if c["type"]=="text"), "")
            payload["messages"][-1]["content"] = text_only
        r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(502, f"OpenRouter: {r.status_code} {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"], TEXT_MODEL

def parse_json(text:str) -> dict:
    text = text.strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    # Find first { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)

def default_actions() -> list:
    return [
        {"type":"double_click","x":0.38,"y":0.70,"wait_ms":2500,"reason":"object grab karo"},
        {"type":"gripper","wait_ms":800,"reason":"close"},
        {"type":"double_click","x":0.55,"y":0.58,"wait_ms":2500,"reason":"target pe lo"},
        {"type":"gripper","wait_ms":800,"reason":"open"},
    ]
