"""
SignSpeak FastAPI Backend  v4
──────────────────────────────
Changes from v3:
  1. Loads StandardScaler (scaler.pkl) saved by train_model.py v4
     Features are scaled before ML inference → major accuracy boost
  2. ML confidence threshold lowered: 55% → 40% (ensemble is well-calibrated)
  3. Label encoder loaded from label_encoder.pkl (fixes class ordering)
  4. Improved motion history: J/Z now use better trajectory analysis
  5. Two-hand support (/classify2) unchanged

Install:
    pip install fastapi uvicorn numpy scikit-learn joblib
    python train_model.py          # builds asl_model.pkl + scaler.pkl
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
import numpy as np
import pathlib, collections
import io, urllib.request

try:
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

app = FastAPI(title="SignSpeak API v4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE    = pathlib.Path(__file__).parent
MODEL_URL  = "https://huggingface.co/SanzharX/signspeak-model/resolve/main/asl_model.pkl"
LE_PATH     = BASE / "label_encoder.pkl"
SCALER_PATH = BASE / "scaler.pkl"

ml_model = None
le       = None
scaler   = None

def load_from_url(url):
    with urllib.request.urlopen(url) as r:
        return joblib.load(io.BytesIO(r.read()))

if SKLEARN_OK:
    try: ml_model = load_from_url(MODEL_URL); print("[SignSpeak] ML model loaded ✓")
    except Exception as ex: print(f"[SignSpeak] Model load failed: {ex}")

    if LE_PATH.exists():
        try: le = joblib.load(LE_PATH); print("[SignSpeak] Label encoder loaded ✓")
        except Exception as ex: print(f"[SignSpeak] Encoder load failed: {ex}")

    if SCALER_PATH.exists():
        try: scaler = joblib.load(SCALER_PATH); print("[SignSpeak] Scaler loaded ✓")
        except Exception as ex: print(f"[SignSpeak] Scaler load failed: {ex}")

HISTORY_LEN = 16
hand_history: dict = {}


# ── Models ────────────────────────────────────────────────────────────────────

class Landmark(BaseModel):
    x: float; y: float; z: float

class HandPayload(BaseModel):
    landmarks: List[Landmark]
    handedness: Optional[str] = "Right"

class TwoHandPayload(BaseModel):
    hands: List[HandPayload]

class SignResult(BaseModel):
    sign: str
    confidence: int
    hint: str
    is_word: bool
    finger_curls: List[float]
    thumb_out: bool
    model_type: str   # "ml" | "rules" | "motion"

class TwoHandResult(BaseModel):
    results: List[SignResult]
    combined: Optional[str] = None


# ── Geometry ──────────────────────────────────────────────────────────────────

def pt(pts, i):
    return np.array([pts[i].x, pts[i].y, pts[i].z], dtype=np.float32)

def d3(a, b): return float(np.linalg.norm(a - b))
def d2(a, b): return float(np.linalg.norm((a - b)[:2]))

def hand_scale(pts):
    return d3(pt(pts, 0), pt(pts, 9)) or 0.001

def finger_curl(pts, tip_i, mcp_i):
    s = hand_scale(pts)
    tip_d = d3(pt(pts, tip_i), pt(pts, 0)) / s
    mcp_d = d3(pt(pts, mcp_i), pt(pts, 0)) / s
    return float(np.clip(2.0 - tip_d / (mcp_d + 1e-6), 0.0, 1.0))

def get_curls(pts):
    return [finger_curl(pts,8,5), finger_curl(pts,12,9),
            finger_curl(pts,16,13), finger_curl(pts,20,17)]

def is_thumb_out(pts):  return d2(pt(pts,4), pt(pts,5)) / hand_scale(pts) > 0.50
def is_thumb_in(pts):   return d2(pt(pts,4), pt(pts,5)) / hand_scale(pts) < 0.35
def is_thumb_side(pts):
    s = hand_scale(pts)
    return 0.35 <= d2(pt(pts,4), pt(pts,5)) / s <= 0.60

def tip_spread(pts, a, b):
    return d2(pt(pts,a), pt(pts,b)) / hand_scale(pts)

def index_up(pts):
    return (pts[5].y - pts[8].y) / hand_scale(pts) > 0.40

def index_sideways(pts):
    s = hand_scale(pts)
    return abs(pts[8].x-pts[5].x)/s > 0.50 and abs(pts[8].y-pts[5].y)/s < 0.40

def index_down(pts):
    return (pts[8].y - pts[5].y) / hand_scale(pts) > 0.30

EXT, CRL = 0.40, 0.60
def e(v): return v < EXT
def c(v): return v > CRL
def m(v): return EXT <= v <= CRL


# ── Feature vector (68-dim) for ML ───────────────────────────────────────────

def make_features(pts) -> np.ndarray:
    wrist = pt(pts, 0)
    s = hand_scale(pts)
    coords = []
    for i in range(21):
        coords.extend(((pt(pts, i) - wrist) / s).tolist())   # 63 normalised xyz
    curls = get_curls(pts)
    to = 1.0 if is_thumb_out(pts) else 0.0
    return np.array(coords + curls + [to], dtype=np.float32)  # 68 total


# ── Motion detector: J, Z, Q ─────────────────────────────────────────────────
#
# All distances are normalised by hand_scale() so the detector works at any
# camera distance.  History stores scale-normalised tip positions so every
# threshold is in "hand-length units" (1.0 ≈ wrist-to-middle-MCP distance).
#
# J shape  : only pinky extended, rest curled, no thumb
#   Motion : tip travels DOWN  (dy > 0.30 hand-units) then hooks
#            LEFT/RIGHT (|dx_hook| > 0.10) in the final 4 frames.
#            Requires consistent downward bias across all frames
#            (median per-frame dy > 0) to reject idle wobble.
#
# Z shape  : only index extended, rest curled (thumb optional)
#   Motion : tip makes exactly 2 direction reversals on the X axis
#            with minimum segment length 0.12 hand-units each,
#            producing the classic top-stroke / diagonal / bottom-stroke.
#            Raw noise (<0.04 units) is ignored before counting reversals.

# Minimum segment size to count as a real stroke (filters jitter)
_Z_MIN_SEG   = 0.12   # hand-units per stroke
_J_MIN_DY    = 0.30   # downward travel for J
_J_HOOK_DX   = 0.10   # hook at the bottom of J


def _normalised_tip_pos(pts, tip_idx: int) -> np.ndarray:
    """Return tip position relative to wrist, divided by hand scale."""
    s = hand_scale(pts)
    return (pt(pts, tip_idx) - pt(pts, 0)) / s


def _count_z_reversals(xs: np.ndarray, min_seg: float) -> int:
    """
    Count meaningful direction reversals in a 1-D position sequence.
    Segments shorter than `min_seg` are treated as noise and skipped.
    Returns number of turns (2 = valid Z: →  ↙  →).
    """
    reversals = 0
    seg_start = xs[0]
    direction = 0   # unknown

    for x in xs[1:]:
        delta = x - seg_start
        if abs(delta) < min_seg:
            continue  # not far enough — skip, keep looking from seg_start
        new_dir = 1 if delta > 0 else -1
        if direction != 0 and new_dir != direction:
            reversals += 1
        direction = new_dir
        seg_start = x

    return reversals


def detect_motion(hist: collections.deque, pts) -> Optional[Tuple[str,int,str]]:
    curls = get_curls(pts)
    ic, mc, rc, pc = curls
    to = is_thumb_out(pts)

    # Hand shapes
    pinky_shape = c(ic) and c(mc) and c(rc) and e(pc) and not to   # J hand
    index_shape = e(ic) and c(mc) and c(rc) and c(pc)               # Z hand
    q_static    = e(ic) and c(mc) and c(rc) and c(pc) and to and index_down(pts)

    # Q is purely static — no history needed
    if q_static:
        return ("Q", 84, "Index down + thumb out = Q")

    # Store scale-normalised tip position + which shape we're in
    tip_idx = 20 if pinky_shape else 8
    norm_pos = _normalised_tip_pos(pts, tip_idx)

    hist.append({
        "pos":       norm_pos,
        "is_pinky":  pinky_shape,
        "is_index":  index_shape,
    })

    if len(hist) < 10:
        return None

    frames = list(hist)   # full deque (max 16 frames)

    # ── J detection ───────────────────────────────────────────────────────────
    # Need at least 10 consecutive pinky-shape frames
    pinky_frames = [f for f in frames if f["is_pinky"]]
    if len(pinky_frames) >= 10:
        positions = np.array([f["pos"] for f in pinky_frames])

        # Total downward travel (Y increases downward in image coords)
        ys  = positions[:, 0]   # NOTE: MediaPipe x maps to horizontal
        # Use actual Y (vertical) for downward motion
        vert = positions[:, 1]
        dy_total = float(vert[-1] - vert[0])

        # Per-frame dy should be mostly positive (moving down)
        frame_dys = np.diff(vert)
        mostly_down = float(np.median(frame_dys)) > 0

        # Hook: last 4 frames should move horizontally (|dx| grows)
        horiz_last = positions[-4:, 0]
        dx_hook = abs(float(horiz_last[-1] - horiz_last[0]))

        if mostly_down and dy_total > _J_MIN_DY and dx_hook > _J_HOOK_DX:
            return ("J", 87, "Pinky sweeps down + hook = J")

    # ── Z detection ───────────────────────────────────────────────────────────
    # Need at least 10 consecutive index-shape frames
    index_frames = [f for f in frames if f["is_index"]]
    if len(index_frames) >= 10:
        positions = np.array([f["pos"] for f in index_frames])
        xs = positions[:, 0]   # horizontal axis = X in normalised coords

        reversals = _count_z_reversals(xs, _Z_MIN_SEG)

        # Z = exactly 2 reversals (right → diagonal-left → right)
        # Allow 2 or 3 to handle slightly sloppy signing
        if reversals in (2, 3):
            return ("Z", 84, "Index traces Z shape")

    return None


# ── Rule-based fallback ───────────────────────────────────────────────────────

def classify_rules(pts, curls, to, ti):
    ic, mc, rc, pc = curls
    ie, me, re, pe = e(ic), e(mc), e(rc), e(pc)
    ib, mb, rb, pb = c(ic), c(mc), c(rc), c(pc)
    ts = is_thumb_side(pts)
    s  = hand_scale(pts)

    # Words
    if to and ie and mb and rb and pe:
        return "LOVE", 93, "Thumb+index+pinky = ILY", True
    if ib and mb and rb and pb and to and (pts[0].y-pts[4].y)/s > 0.30:
        return "HELP", 91, "Thumbs up = HELP", True
    if ie and me and re and pe and to and tip_spread(pts,8,20) > 0.30:
        return "HELLO", 90, "Open hand = HELLO", True

    # B: all 4 up, thumb in
    if ie and me and re and pe and ti:
        return "B", 92, "4 fingers up thumb in = B", False

    # W: 3 fingers up
    if ie and me and re and pb and not to:
        return "W", 87, "3 fingers up = W", False

    # K: index+middle + thumb out
    if ie and me and rb and pb and to:
        return "K", 82, "Index+middle+thumb = K", False

    # L: only index + thumb up
    if ie and mb and rb and pb and to:
        return "L", 91, "Index+thumb L = L", False

    # R: crossed (spread < 0.18)
    if ie and me and rb and pb and not to and tip_spread(pts,8,12) < 0.18:
        return "R", 80, "Crossed fingers = R", False

    # U: index+middle close
    if ie and me and rb and pb and not to and tip_spread(pts,8,12) < 0.28:
        return "U", 88, "Index+middle together = U", False

    # V: index+middle spread
    if ie and me and rb and pb and not to:
        return "V", 90, "Peace sign = V", False

    # H: index+middle sideways
    if ie and me and rb and pb and not to and not index_up(pts):
        return "H", 82, "Index+middle sideways = H", False

    # G: index sideways + thumb side
    if ie and mb and rb and pb and index_sideways(pts):
        return "G", 81, "Index sideways = G", False

    # D: index up only
    if ie and mb and rb and pb and not to:
        return "D", 86, "Index up = D", False

    # Y: pinky+thumb out
    if ib and mb and rb and pe and to:
        return "Y", 89, "Thumb+pinky = Y", False

    # I: pinky only
    if ib and mb and rb and pe and not to:
        return "I", 89, "Pinky only = I", False

    # F: index half-bent, others up
    if m(ic) and me and re and pe and to:
        return "F", 83, "Pinch+3up = F", False

    # X: hooked index
    if m(ic) and mb and rb and pb and not to:
        return "X", 80, "Hooked index = X", False

    # P: index down + thumb out
    if e(ic) and c(mc) and c(rc) and c(pc) and to and index_down(pts):
        return "P", 80, "Index down+thumb = P", False

    # O: all mid-bent, thumb in
    if m(ic) and m(mc) and m(rc) and m(pc) and ti:
        return "O", 82, "All form O", False

    # C: all mid-bent, thumb out
    if m(ic) and m(mc) and m(rc) and m(pc) and not ti:
        return "C", 85, "Curved C", False

    # N: index+middle over thumb
    if ib and mb and re and pe and not to:
        return "N", 78, "Index+middle over = N", False

    # E: all deeply curled
    if ic>0.70 and mc>0.70 and rc>0.70 and pc>0.70 and ti:
        return "E", 84, "All curled = E", False

    # S: fist + thumb over (side)
    if ib and mb and rb and pb and ts and pts[4].y > pts[8].y:
        return "S", 86, "Fist thumb over = S", False

    # T: fist + thumb between fingers
    if ib and mb and rb and pb and not to:
        if d3(pt(pts,4), pt(pts,6)) / s < 0.30:
            return "T", 77, "Thumb between = T", False

    # M: fist + thumb in
    if ib and mb and rb and pb and ti:
        return "M", 79, "3 over thumb = M", False

    # A: fist (catch-all)
    if ib and mb and rb and pb:
        return "A", 87, "Fist = A", False

    return "?", 0, "Hold a clear ASL shape", False


# ── Unified classify ──────────────────────────────────────────────────────────

HINT_MAP = {
    "A":"Fist","B":"4 fingers up","C":"Curved C","D":"Index up","E":"All curled",
    "F":"Pinch+3up","G":"Index sideways","H":"2 fingers side","I":"Pinky up",
    "J":"Pinky hook (motion)","K":"2+thumb","L":"L shape","M":"3 over thumb",
    "N":"2 over thumb","O":"O shape","P":"Index down+thumb","Q":"Index down out",
    "R":"Crossed","S":"Fist over","T":"Thumb between","U":"2 together",
    "V":"Peace","W":"3 up","X":"Hook","Y":"Thumb+pinky","Z":"Index traces Z",
    "LOVE":"ILY sign","HELP":"Thumbs up","HELLO":"Open hand",
}
WORD_SIGNS = {"LOVE","HELP","HELLO"}

# Lowered from 55 → 40: ensemble model is well-calibrated, trust it more
ML_CONFIDENCE_THRESHOLD = 40

def classify_hand(pts, handedness="Right") -> SignResult:
    if len(pts) != 21:
        return SignResult(sign="?", confidence=0, hint="Need 21 landmarks",
                          is_word=False, finger_curls=[0]*4, thumb_out=False, model_type="none")

    curls = get_curls(pts)
    to    = is_thumb_out(pts)
    ti    = is_thumb_in(pts)

    if handedness not in hand_history:
        hand_history[handedness] = collections.deque(maxlen=HISTORY_LEN)
    hist = hand_history[handedness]

    # 1. Motion (J, Z, Q)
    motion = detect_motion(hist, pts)
    if motion:
        sign, conf, hint = motion
        return SignResult(sign=sign, confidence=conf, hint=hint,
                          is_word=False, finger_curls=curls, thumb_out=to, model_type="motion")

    # 2. ML model
    if ml_model is not None:
        try:
            feat = make_features(pts).reshape(1, -1)

            # Apply scaler if available (trained with v4)
            if scaler is not None:
                feat = scaler.transform(feat)

            proba = ml_model.predict_proba(feat)[0]
            idx   = int(np.argmax(proba))
            conf  = int(proba[idx] * 100)

            # Decode using label encoder if available
            if le is not None:
                pred = le.classes_[idx]
            else:
                pred = ml_model.classes_[idx]

            if conf >= ML_CONFIDENCE_THRESHOLD:
                return SignResult(
                    sign=pred, confidence=conf,
                    hint=HINT_MAP.get(pred, pred),
                    is_word=pred in WORD_SIGNS,
                    finger_curls=curls, thumb_out=to, model_type="ml"
                )
        except Exception as ex:
            print(f"[ML] {ex}")

    # 3. Rules fallback
    sign, conf, hint, is_word = classify_rules(pts, curls, to, ti)
    return SignResult(sign=sign, confidence=conf, hint=hint,
                      is_word=is_word, finger_curls=curls, thumb_out=to, model_type="rules")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=SignResult)
async def ep_classify(payload: HandPayload):
    return classify_hand(payload.landmarks, payload.handedness or "Right")


@app.post("/classify2", response_model=TwoHandResult)
async def ep_classify2(payload: TwoHandPayload):
    results = [classify_hand(h.landmarks, h.handedness or f"Hand{i}")
               for i, h in enumerate(payload.hands[:2])]
    combined = None
    if len(results) == 2:
        a, b = results[0].sign, results[1].sign
        if a not in ("?","") and b not in ("?","") and a != b:
            combined = f"{a} {b}"
    return TwoHandResult(results=results, combined=combined)


@app.get("/health")
async def health():
    mode = "ml+motion+rules" if ml_model else "motion+rules (no model)"
    return {
        "status": "ok",
        "model": f"SignSpeak v4 ({mode})",
        "ml_loaded": ml_model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": le is not None,
        "ml_threshold": ML_CONFIDENCE_THRESHOLD,
    }


@app.get("/")
async def root():
    return {"name":"SignSpeak API v4","POST /classify":"single hand","POST /classify2":"two hands"}
