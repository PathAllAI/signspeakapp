"""
SignSpeak FastAPI Backend
─────────────────────────
Receives 21 hand landmarks from the frontend,
runs a robust rule-based + angle classifier,
returns the detected sign + confidence.

Run:
    pip install fastapi uvicorn numpy
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import math

app = FastAPI(title="SignSpeak API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data models ───────────────────────────────────────────────────────────────

class Landmark(BaseModel):
    x: float
    y: float
    z: float

class HandPayload(BaseModel):
    landmarks: List[Landmark]   # always 21 points
    handedness: Optional[str] = "Right"

class SignResult(BaseModel):
    sign: str
    confidence: int
    hint: str
    is_word: bool
    finger_curls: List[float]   # [index, middle, ring, pinky] 0=ext 1=curl
    thumb_out: bool

# ── Geometry helpers ──────────────────────────────────────────────────────────

def lm(pts, i):
    return np.array([pts[i].x, pts[i].y, pts[i].z])

def dist3(a, b):
    return float(np.linalg.norm(a - b))

def dist2(a, b):
    return float(np.linalg.norm(a[:2] - b[:2]))

def hand_scale(pts):
    """Wrist → middle-MCP distance as normalisation factor."""
    return dist3(lm(pts, 0), lm(pts, 9)) or 0.001

def finger_curl(pts, tip_i, mcp_i):
    """
    0 = fully extended, 1 = fully curled.
    Uses tip-to-wrist / mcp-to-wrist ratio, scale-invariant.
    """
    scale = hand_scale(pts)
    tip_d = dist3(lm(pts, tip_i), lm(pts, 0)) / scale
    mcp_d = dist3(lm(pts, mcp_i), lm(pts, 0)) / scale
    ratio = tip_d / (mcp_d + 1e-6)
    return float(np.clip(2.0 - ratio, 0.0, 1.0))

def thumb_out(pts):
    scale = hand_scale(pts)
    return dist2(lm(pts, 4), lm(pts, 5)) / scale > 0.50

def thumb_in(pts):
    scale = hand_scale(pts)
    return dist2(lm(pts, 4), lm(pts, 5)) / scale < 0.35

def tip_spread(pts, a, b):
    return dist2(lm(pts, a), lm(pts, b)) / hand_scale(pts)

def index_points_up(pts):
    scale = hand_scale(pts)
    return (pts[5].y - pts[8].y) / scale > 0.40

def index_points_sideways(pts):
    scale = hand_scale(pts)
    return (abs(pts[8].x - pts[5].x) / scale > 0.50 and
            abs(pts[8].y - pts[5].y) / scale < 0.40)

def get_curls(pts):
    """Returns [index_curl, middle_curl, ring_curl, pinky_curl]"""
    return [
        finger_curl(pts, 8,  5),   # index
        finger_curl(pts, 12, 9),   # middle
        finger_curl(pts, 16, 13),  # ring
        finger_curl(pts, 20, 17),  # pinky
    ]

# Thresholds
EXT = 0.40   # below = extended
CRL = 0.60   # above = curled

def e(v): return v < EXT
def c(v): return v > CRL
def m(v): return EXT <= v <= CRL   # mid / half-bent

# ── Main classifier ───────────────────────────────────────────────────────────

def classify(pts: List[Landmark]) -> SignResult:
    curls = get_curls(pts)
    ic, mc, rc, pc = curls
    ie, me, re, pe = e(ic), e(mc), e(rc), e(pc)
    ib, mb, rb, pb = c(ic), c(mc), c(rc), c(pc)
    to = thumb_out(pts)
    ti = thumb_in(pts)
    scale = hand_scale(pts)

    def r(sign, conf, hint, is_word=False):
        return SignResult(
            sign=sign, confidence=conf, hint=hint,
            is_word=is_word, finger_curls=curls, thumb_out=to
        )

    # ── Common ASL words ──────────────────────────────────────────────────────

    # ILY / LOVE: thumb + index + pinky extended, middle + ring curled
    if to and ie and mb and rb and pe:
        return r("LOVE", 93, "Thumb + index + pinky out — ILY", is_word=True)

    # HELP / Thumbs-up: all fingers curled, thumb pointing up (tip above wrist)
    if ib and mb and rb and pb and to and (pts[0].y - pts[4].y) / scale > 0.30:
        return r("HELP", 91, "Thumbs up — HELP / YES", is_word=True)

    # HELLO / Open hand: all extended, spread wide
    if ie and me and re and pe and to and tip_spread(pts, 8, 20) > 0.30:
        return r("HELLO", 90, "Open hand — wave HELLO", is_word=True)

    # ── ASL A–Z letters ───────────────────────────────────────────────────────

    # A: fist, thumb tucked to side
    if ib and mb and rb and pb and ti:
        return r("A", 87, "Fist, thumb at side")

    # B: all four fingers up, thumb folded in
    if ie and me and re and pe and ti:
        return r("B", 92, "Four fingers up, thumb in")

    # C: all fingers half-bent in C curve
    if m(ic) and m(mc) and m(rc) and m(pc) and not ti:
        return r("C", 85, "Curved C shape")

    # D: index up, rest curl toward thumb
    if ie and mb and rb and pb and not to:
        return r("D", 86, "Index up, others curl to thumb")

    # E: all deeply curled, thumb tucked under
    if ic > 0.70 and mc > 0.70 and rc > 0.70 and pc > 0.70 and ti:
        return r("E", 84, "All fingers tightly curled")

    # F: index half-bent (pinching thumb), others up
    if m(ic) and me and re and pe and to:
        return r("F", 83, "Index+thumb pinch, others up")

    # G: index pointing sideways, rest curled
    if ie and mb and rb and pb and index_points_sideways(pts):
        return r("G", 81, "Index points sideways")

    # H: index+middle pointing sideways together
    if ie and me and rb and pb and not to and not index_points_up(pts):
        return r("H", 82, "Index+middle horizontal side-by-side")

    # I: only pinky up
    if ib and mb and rb and pe and not to:
        return r("I", 89, "Pinky only extended")

    # K: index+middle up, thumb between them (out)
    if ie and me and rb and pb and to:
        return r("K", 81, "Index+middle up, thumb between")

    # L: index up + thumb out, others curled
    if ie and mb and rb and pb and to:
        return r("L", 91, "L-shape: index up + thumb out")

    # M: index+middle+ring folded over thumb
    if ib and mb and rb and pb and not to:
        return r("M", 79, "Three fingers fold over thumb")

    # N: index+middle fold over thumb (ring+pinky up)
    if ib and mb and re and pe and not to:
        return r("N", 78, "Index+middle over thumb, others up")

    # O: all fingers form O touching thumb
    if m(ic) and m(mc) and m(rc) and m(pc) and ti:
        return r("O", 82, "All fingers form O shape")

    # R: index+middle up and crossed (very close tips)
    if ie and me and rb and pb and not to and tip_spread(pts, 8, 12) < 0.25:
        return r("R", 80, "Index+middle crossed")

    # S: fist, thumb over fingers
    if ib and mb and rb and pb and to and pts[4].y > pts[8].y:
        return r("S", 86, "Fist, thumb over fingers")

    # T: thumb tucked between index and middle
    if ib and mb and rb and pb and not to:
        d = dist3(lm(pts, 4), lm(pts, 6)) / scale
        if d < 0.30:
            return r("T", 77, "Thumb between index and middle")

    # U: index+middle up, close together
    if ie and me and rb and pb and not to and tip_spread(pts, 8, 12) < 0.30:
        return r("U", 88, "Index+middle up together")

    # V: index+middle up, spread apart
    if ie and me and rb and pb and not to and tip_spread(pts, 8, 12) >= 0.30:
        return r("V", 90, "V sign — fingers spread")

    # W: index+middle+ring up
    if ie and me and re and pb and not to:
        return r("W", 87, "Three fingers up — W")

    # X: only index half-hooked
    if m(ic) and mb and rb and pb and not to:
        return r("X", 80, "Hooked index finger")

    # Y: thumb + pinky out, rest curled
    if ib and mb and rb and pe and to:
        return r("Y", 89, "Thumb + pinky out — Y")

    return r("?", 0, "Hold a clear ASL shape steady")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=SignResult)
async def classify_hand(payload: HandPayload):
    if len(payload.landmarks) != 21:
        return SignResult(sign="?", confidence=0,
                          hint="Expected 21 landmarks",
                          is_word=False, finger_curls=[0,0,0,0], thumb_out=False)
    return classify(payload.landmarks)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "rule-based curl classifier v2"}


@app.get("/")
async def root():
    return {
        "name": "SignSpeak API",
        "endpoints": {
            "POST /classify": "Send 21 landmarks, get sign back",
            "GET  /health":   "Health check"
        }
    }