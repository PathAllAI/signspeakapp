# SignSpeak — Setup Guide

## Folder structure
```
signspeakapp/
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/
    └── index.html
```

## 1 — Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Verify it works: open http://localhost:8000 in your browser.

## 2 — Open the frontend

Just open `frontend/index.html` directly in **Chrome** or **Edge**.
(Firefox blocks camera on file:// — use Chrome)

## 3 — Connect

1. Click **Enable Camera**
2. The app auto-tests the backend at `http://localhost:8000`
3. If the backend dot turns green → you're connected
4. Show your hand → hold a gesture 1 second → letter/word confirms

## How it works

```
Camera (video)
    ↓
MediaPipe Hands (browser WASM)
    ↓ 21 landmark coordinates (x,y,z) — NOT video
    ↓ ~10 requests/sec  (~1KB each)
FastAPI backend /classify
    ↓ finger curl ratios + geometry
    ↓ returns: sign, confidence, hint, finger_curls
Frontend
    ↓ draws dots on canvas, fills hold bar
    ↓ confirmed → letter buffer → word → sentence → TTS
```

## Upgrading to ML later

Replace the rule-based `classify()` function in `main.py` with:
- Load a trained sklearn / TFLite model at startup
- Pass the 63 floats (21 landmarks × xyz) as a feature vector
- Train on: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Example swap-in:
```python
import joblib
model = joblib.load("asl_model.pkl")   # trained RandomForest / SVM

@app.post("/classify")
async def classify_hand(payload: HandPayload):
    features = [[l.x, l.y, l.z] for l in payload.landmarks]
    flat = [v for xyz in features for v in xyz]   # 63 floats
    pred = model.predict([flat])[0]
    prob = model.predict_proba([flat]).max()
    return SignResult(sign=pred, confidence=int(prob*100), ...)
```
