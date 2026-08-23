# Axis AI Bot — Vercel Backend

FastAPI backend jo Axis Robotics tasks ke liye AI action plans generate karta hai.

---

## Deploy Kaise Karein (5 Steps)

### Step 1 — GitHub pe push karo
```bash
git init
git add .
git commit -m "Axis AI Bot backend"
git remote add origin https://github.com/AAPKA_USERNAME/axis-backend.git
git push -u origin main
```

### Step 2 — Vercel pe jao
1. [vercel.com](https://vercel.com) open karo
2. **"Add New Project"** click karo
3. GitHub repo import karo (`axis-backend`)
4. Framework: **Other** select karo

### Step 3 — Environment Variable set karo
Vercel dashboard → Project → **Settings → Environment Variables**:
```
Name:  OPENROUTER_API_KEY
Value: sk-or-v1-xxxxxxxx   ← openrouter.ai/keys se lo
```

### Step 4 — Deploy!
**"Deploy"** button dabao — 1-2 minute mein live ho jaayega.

### Step 5 — URL copy karo
Deploy hone ke baad URL milega:
```
https://axis-backend-xyz.vercel.app
```
Yeh URL **browser extension ke popup mein** daalo.

---

## Local Test Kaise Karein

```bash
pip install -r requirements.txt
cp .env.example .env    # API key daalo
uvicorn main:app --reload
```

Browser mein: `http://localhost:8000`

### Test API Call:
```bash
curl -X POST http://localhost:8000/api/plan \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "Pick and Place Cube",
    "steps": ["Pick up the red cube", "Place it on the blue plate"],
    "difficulty": "Easy"
  }'
```

### Expected Response:
```json
{
  "actions": [
    {"type": "double_click", "x": 0.35, "y": 0.65, "wait_ms": 2000, "reason": "grab cube"},
    {"type": "gripper", "wait_ms": 800, "reason": "close gripper"},
    ...
  ],
  "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
  "task": "Pick and Place Cube"
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status check |
| `/health` | GET | Health check |
| `/api/plan` | POST | Action plan generate karo |

---

## Architecture

```
Browser Extension
      ↓ POST /api/plan
Vercel Backend (FastAPI)
      ↓
OpenRouter API
      ↓
nvidia/nemotron-3-ultra-550b-a55b:free
      ↓ JSON actions
Vercel Backend
      ↓
Browser Extension → Robot Control
```
