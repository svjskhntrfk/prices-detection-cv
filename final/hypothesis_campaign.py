#!/usr/bin/env python3
"""Final OCR orchestrator locked to one champion bundle.

Public CLI is intentionally minimal:
  - run_bundle
  - show_state
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOK = "notebookc9d692d630.ipynb"
DEFAULT_OUTPUT_ROOT = "./repro_outputs"
DEFAULT_SAMPLE_SIZE = 96
DEFAULT_VISUAL_PANEL_SIZE = 24
CHAMPION_BUNDLE_NAME = "omega2_fixed"
CHAMPION_BUNDLE_TAGS = [
    "data_input/H2",
    "preprocess/H1",
    "ocr/H4",
    "parsers/H2",
    "parsers/H4",
    "qr_barcode/H1",
    "track_merge/H1",
]
KEY_FIELDS = ["product_name", "price_discount", "barcode"]
KEY_WEIGHTS = {"product_name": 0.45, "price_discount": 0.35, "barcode": 0.20}
PRICE_FIELDS = ["price_default", "price_card", "price_discount"]
NONE_VALUES = {"", "нет", "none", "null", "nan"}
CASE_PROXY_V2_COMPONENT_WEIGHTS = {
    "name_quality": 0.35,
    "price_quality": 0.30,
    "barcode_quality": 0.20,
    "value_validity": 0.15,
}


@dataclass(frozen=True)
class Hypothesis:
    stage: str
    hid: str
    title: str
    summary: str


CATALOG: dict[str, list[Hypothesis]] = {
    "data_input": [
        Hypothesis("data_input", "H2", "Stable frame normalization", "Deterministic frame ordering with rank-aware preference."),
    ],
    "preprocess": [
        Hypothesis("preprocess", "H1", "Adaptive OCR variants", "Start with original+2x and escalate to heavier variants only when needed."),
    ],
    "ocr": [
        Hypothesis("ocr", "H4", "Multi-ROI OCR ensemble", "Adaptive full+ROI OCR with token clustering and price-zone fallback."),
    ],
    "qr_barcode": [
        Hypothesis("qr_barcode", "H1", "Barcode validation", "Accept only format-valid barcode candidates first."),
    ],
    "parsers": [
        Hypothesis("parsers", "H2", "Card vs discount anchors", "Improve disambiguation of price_card vs price_discount."),
        Hypothesis("parsers", "H4", "Price recovery v2", "Recover split ruble/kopeck tokens and enforce strict price validity."),
    ],
    "track_merge": [
        Hypothesis("track_merge", "H1", "Confidence-aware merge", "Prefer confidence+validity over pure frequency for key fields."),
    ],
}


def _catalog_index() -> dict[tuple[str, str], Hypothesis]:
    idx: dict[tuple[str, str], Hypothesis] = {}
    for stage, items in CATALOG.items():
        for h in items:
            idx[(stage, h.hid)] = h
    return idx


HYP_INDEX = _catalog_index()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def as_markdown_row(values: list[str]) -> str:
    escaped = [str(v).replace("|", "\\|") for v in values]
    return "| " + " | ".join(escaped) + " |"


def is_non_empty_value(v: Any) -> bool:
    s = "" if v is None else str(v).strip().lower()
    return s not in NONE_VALUES


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_price_value(raw: Any) -> float | None:
    s = "" if raw is None else str(raw).strip().lower()
    if s in NONE_VALUES:
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^\d.\s]", "", s)
    s = re.sub(r"\s+", "", s)
    if not s or s.count(".") > 1:
        return None
    if re.fullmatch(r"\d{3,7}", s):
        s = f"{s[:-2]}.{s[-2:]}"
    if not re.fullmatch(r"\d+(?:\.\d{1,2})?", s):
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not (0 < v < 1_000_000):
        return None
    return v


def _name_plausibility(raw: Any) -> float:
    s = "" if raw is None else str(raw).strip()
    if not s:
        return 0.0
    alnum = re.sub(r"[^\wа-яА-ЯёЁ]", "", s, flags=re.UNICODE)
    letters = re.findall(r"[a-zA-Zа-яА-ЯёЁ]", s)
    digits = re.findall(r"\d", s)
    if not letters:
        return 0.0
    token_count = len([t for t in re.split(r"\s+", s) if t.strip()])
    alpha_ratio = len(letters) / max(1, len(alnum))
    digit_ratio = len(digits) / max(1, len(alnum))
    length_score = 1.0 if 6 <= len(s) <= 96 else 0.65
    token_score = 1.0 if token_count >= 2 else 0.6
    alpha_score = 1.0 if alpha_ratio >= 0.55 else alpha_ratio / 0.55
    digit_penalty = 1.0 - clamp01((digit_ratio - 0.20) / 0.40)
    return clamp01(0.35 * token_score + 0.25 * length_score + 0.30 * alpha_score + 0.10 * digit_penalty)


def _ean_checksum_ok(num: str) -> bool:
    if not num.isdigit():
        return False
    if len(num) == 13:
        body, check = num[:-1], int(num[-1])
        s = 0
        for i, ch in enumerate(body):
            d = int(ch)
            s += d if i % 2 == 0 else 3 * d
        return ((10 - (s % 10)) % 10) == check
    if len(num) == 8:
        body, check = num[:-1], int(num[-1])
        s = 0
        for i, ch in enumerate(body):
            d = int(ch)
            s += 3 * d if i % 2 == 0 else d
        return ((10 - (s % 10)) % 10) == check
    return False


def _barcode_quality(raw: Any) -> float:
    s = "" if raw is None else str(raw).strip()
    if not s:
        return 0.0
    digits = re.sub(r"\D", "", s)
    if not digits:
        return 0.0
    if len(digits) in {8, 13} and _ean_checksum_ok(digits):
        return 1.0
    if len(digits) in {12, 14}:
        return 0.85
    if len(digits) in {8, 13}:
        return 0.55
    if 6 <= len(digits) <= 16:
        return 0.25
    return 0.10


def _price_row_quality(row: dict[str, Any]) -> float:
    raw_values = [row.get(f, "") for f in PRICE_FIELDS]
    present = [v for v in raw_values if is_non_empty_value(v)]
    if not present:
        return 0.0

    parsed = {f: _parse_price_value(row.get(f, "")) for f in PRICE_FIELDS}
    valid_count = sum(1 for v in parsed.values() if v is not None)
    score = 0.55 * (valid_count / len(PRICE_FIELDS))

    any_valid = [v for v in parsed.values() if v is not None]
    if any_valid:
        score += 0.20

    p_default = parsed["price_default"]
    p_card = parsed["price_card"]
    p_discount = parsed["price_discount"]
    if p_default is not None and p_discount is not None and p_discount <= p_default:
        score += 0.15
    if p_default is not None and p_card is not None and p_card <= p_default * 1.05:
        score += 0.10

    invalid_present = sum(1 for f in PRICE_FIELDS if is_non_empty_value(row.get(f, "")) and parsed[f] is None)
    score -= 0.08 * invalid_present
    return clamp01(score)


def _row_value_validity(row: dict[str, Any]) -> float:
    checks: list[float] = []
    name_raw = row.get("product_name", "")
    if is_non_empty_value(name_raw):
        checks.append(1.0 if _name_plausibility(name_raw) >= 0.50 else 0.0)

    price_present = any(is_non_empty_value(row.get(f, "")) for f in PRICE_FIELDS)
    if price_present:
        checks.append(1.0 if any(_parse_price_value(row.get(f, "")) is not None for f in PRICE_FIELDS) else 0.0)

    barcode_raw = row.get("barcode", "")
    if is_non_empty_value(barcode_raw):
        checks.append(1.0 if _barcode_quality(barcode_raw) >= 0.55 else 0.0)

    discount_raw = "" if row.get("discount_amount", None) is None else str(row.get("discount_amount", "")).strip().lower()
    if discount_raw and discount_raw not in NONE_VALUES:
        checks.append(1.0 if bool(re.fullmatch(r"\d{1,2}\s*%", discount_raw)) else 0.0)

    if not checks:
        return 0.0
    return clamp01(sum(checks) / len(checks))


def _score_metric_components(rows: list[dict[str, str]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "name_quality": 0.0,
            "price_quality": 0.0,
            "barcode_quality": 0.0,
            "value_validity": 0.0,
        }

    name_quality = sum(_name_plausibility(r.get("product_name", "")) for r in rows) / total
    price_quality = sum(_price_row_quality(r) for r in rows) / total
    barcode_quality = sum(_barcode_quality(r.get("barcode", "")) for r in rows) / total
    value_validity = sum(_row_value_validity(r) for r in rows) / total
    return {
        "name_quality": clamp01(name_quality),
        "price_quality": clamp01(price_quality),
        "barcode_quality": clamp01(barcode_quality),
        "value_validity": clamp01(value_validity),
    }


def compute_proxy_metrics(csv_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(csv_path)
    total = len(rows)
    if total == 0:
        return {
            "rows": 0,
            "fill_rate": {},
            "key_fill": {k: 0.0 for k in KEY_FIELDS},
            "proxy_score": 0.0,
            "proxy_score_v1": 0.0,
            "price_any_fill": 0.0,
            "component_scores": {k: 0.0 for k in CASE_PROXY_V2_COMPONENT_WEIGHTS},
            "case_proxy_v2": 0.0,
            "ranking_mode": "case_proxy_v2",
            "ranking_score": 0.0,
        }

    fill_rate: dict[str, float] = {}
    headers = list(rows[0].keys())
    for col in headers:
        cnt = sum(1 for r in rows if is_non_empty_value(r.get(col, "")))
        fill_rate[col] = cnt / total

    key_fill = {k: fill_rate.get(k, 0.0) for k in KEY_FIELDS}
    proxy_score = sum(key_fill[k] * KEY_WEIGHTS[k] for k in KEY_FIELDS)

    price_any_fill = sum(
        1 for r in rows if any(is_non_empty_value(r.get(f, "")) for f in PRICE_FIELDS)
    ) / total

    components = _score_metric_components(rows)
    case_proxy_v2 = 0.0
    for key, weight in CASE_PROXY_V2_COMPONENT_WEIGHTS.items():
        case_proxy_v2 += weight * float(components.get(key, 0.0))
    case_proxy_v2 = clamp01(case_proxy_v2)

    return {
        "rows": total,
        "fill_rate": fill_rate,
        "key_fill": key_fill,
        "proxy_score": proxy_score,
        "proxy_score_v1": proxy_score,
        "price_any_fill": price_any_fill,
        "component_scores": components,
        "case_proxy_v2": case_proxy_v2,
        "ranking_mode": "case_proxy_v2",
        "ranking_score": case_proxy_v2,
    }


def discover_tracks(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.glob("track_*") if p.is_dir()])


def pick_sample_tracks(all_tracks: list[Path], sample_size: int, seed: int) -> list[str]:
    ordered = sorted(all_tracks, key=lambda p: sha1_short(f"{seed}:{p.name}"))
    return [p.name for p in ordered[: max(1, min(sample_size, len(ordered)))]]


def pick_visual_panel(sample_names: list[str], visual_size: int, seed: int) -> list[str]:
    ordered = sorted(sample_names, key=lambda n: sha1_short(f"panel:{seed}:{n}"))
    return ordered[: max(1, min(visual_size, len(ordered)))]


def hypothesis_patch(stage: str, hid: str) -> str:
    key = (stage, hid)
    if key not in HYP_INDEX:
        raise ValueError(f"Unknown hypothesis: {stage}/{hid}")

    # NOTE: this code is injected into notebook runtime before main processing loop.
    patches: dict[tuple[str, str], str] = {}

    patches[("data_input", "H2")] = r'''

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def _frame_sort_key(p: Path) -> tuple[int, int, int, str]:
    name = p.name.lower()
    m_rank = re.search(r"rank(\d+)", name)
    m_idx = re.search(r"_(\d+)", name)
    rank = int(m_rank.group(1)) if m_rank else 9999
    idx = int(m_idx.group(1)) if m_idx else 999999
    return (0 if m_rank else 1, rank, idx, name)

def collect_track_images(track_dir: Path) -> list[Path]:
    imgs = [p for p in track_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")]
    imgs.sort(key=_frame_sort_key)
    return imgs
print("[HYP data_input/H2] rank-aware stable frame ordering enabled")
'''

    patches[("preprocess", "H1")] = r'''

_prev_run_ocr_on_image_variants = run_ocr_on_image_variants

def run_ocr_on_image_variants(image_bgr: np.ndarray) -> dict[str, Any]:
    variants = make_variants(image_bgr)
    primary = ["original", "upscale_2x"]
    fallback = ["clahe", "upscale_4x"]

    all_tokens = []
    by_variant = {}

    for name in primary:
        vimg = variants.get(name)
        if vimg is None:
            continue
        vt = run_paddle_ocr(vimg, variant_name=name)
        by_variant[name] = vt
        all_tokens.extend(vt)

    dedup = deduplicate_tokens(all_tokens)
    need_escalation = len(dedup) < 6 or not any(looks_like_price_text(t.get("text", "")) for t in dedup)
    if need_escalation:
        for name in fallback:
            vimg = variants.get(name)
            if vimg is None:
                continue
            vt = run_paddle_ocr(vimg, variant_name=name)
            by_variant[name] = vt
            all_tokens.extend(vt)
        dedup = deduplicate_tokens(all_tokens)

    return {
        "tokens": dedup,
        "tokens_all": all_tokens,
        "tokens_by_variant": by_variant,
        "ocr_backend": OCR_BACKEND_NAME if OCR_ENGINE is not None else "none",
    }

print("[HYP preprocess/H1] adaptive variant escalation enabled")
'''

    patches[("ocr", "H4")] = r'''

_prev_run_ocr_on_image_variants = run_ocr_on_image_variants

def _token_area(tok: dict[str, Any]) -> int:
    b = list(tok.get("bbox", [0, 0, 0, 0]))
    if len(b) != 4:
        return 0
    return max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1]))


def _shift_token_bbox(tok: dict[str, Any], dx: int, dy: int, roi_tag: str) -> dict[str, Any]:
    t = dict(tok)
    b = list(t.get("bbox", [0, 0, 0, 0]))
    if len(b) == 4:
        b[0] += dx
        b[1] += dy
        b[2] += dx
        b[3] += dy
    t["bbox"] = b
    t["roi"] = roi_tag
    t["variant"] = f"{roi_tag}:{t.get('variant', '')}"
    return t


def _cluster_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for tok in tokens:
        txt = normalize_whitespace(str(tok.get("text", "")))
        if not txt:
            continue
        t = dict(tok)
        t["text"] = txt
        b = list(t.get("bbox", [0, 0, 0, 0]))
        if len(b) != 4:
            b = [0, 0, 0, 0]
            t["bbox"] = b
        yc = (int(b[1]) + int(b[3])) // 2
        band = yc // 48
        key = (normalized_text_key(txt), band)
        conf = float(t.get("confidence", 0.0))
        area = max(1, _token_area(t))
        price_bonus = 0.12 if looks_like_price_text(txt) else 0.0
        score = conf + 0.0015 * min(area, 6000) + price_bonus
        prev = best.get(key)
        if prev is None or score > prev[0]:
            best[key] = (score, t)
    out = [v[1] for v in best.values()]
    out.sort(key=lambda t: (t.get("bbox", [0, 0, 0, 0])[1], t.get("bbox", [0, 0, 0, 0])[0]))
    return out


def run_ocr_on_image_variants(image_bgr: np.ndarray) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    rois = [
        ("full", 0, 0, w, h),
        ("top", 0, 0, w, max(1, int(h * 0.62))),
        ("bottom", 0, int(h * 0.35), w, h),
        ("center", int(w * 0.08), int(h * 0.12), int(w * 0.92), int(h * 0.88)),
        ("barcode", 0, int(h * 0.55), w, h),
    ]

    all_tokens = []
    tokens_by_variant = {}
    backend = "none"

    for tag, x1, y1, x2, y2 in rois:
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rr = _prev_run_ocr_on_image_variants(roi)
        backend = rr.get("ocr_backend", backend)

        shifted_main = [_shift_token_bbox(t, x1, y1, tag) for t in rr.get("tokens", [])]
        all_tokens.extend(shifted_main)

        for vname, vtoks in rr.get("tokens_by_variant", {}).items():
            key = f"{tag}:{vname}"
            tokens_by_variant[key] = [_shift_token_bbox(t, x1, y1, tag) for t in vtoks]

    clustered = _cluster_tokens(all_tokens)
    need_price_retry = len(clustered) < 6 or not any(looks_like_price_text(t.get("text", "")) for t in clustered)
    if need_price_retry:
        y0 = int(h * 0.45)
        roi = image_bgr[y0:h, 0:w]
        if roi.size > 0:
            rr = _prev_run_ocr_on_image_variants(roi)
            extra = [_shift_token_bbox(t, 0, y0, "retry") for t in rr.get("tokens", [])]
            all_tokens.extend(extra)
            for vname, vtoks in rr.get("tokens_by_variant", {}).items():
                key = f"retry:{vname}"
                tokens_by_variant[key] = [_shift_token_bbox(t, 0, y0, "retry") for t in vtoks]
            clustered = _cluster_tokens(all_tokens)

    return {
        "tokens": clustered,
        "tokens_all": all_tokens,
        "tokens_by_variant": tokens_by_variant,
        "ocr_backend": backend,
    }

print("[HYP ocr/H4] multi-ROI OCR ensemble enabled")
'''

    patches[("parsers", "H2")] = r'''

_prev_parse_prices = parse_prices

def parse_prices(tokens: list[dict[str, Any]], color: str, special_symbols: str) -> dict[str, str]:
    res = _prev_parse_prices(tokens, color=color, special_symbols=special_symbols)
    txt = joined_text(tokens).lower()

    # Context anchors to separate card vs discount
    has_card_anchor = any(k in txt for k in ["карта", "по карте", "card"])
    has_discount_anchor = any(k in txt for k in ["скид", "акц", "%", "выгод"])

    if has_card_anchor and res.get("price_card") in {"", "нет"} and res.get("price_discount") not in {"", "нет"}:
        res["price_card"] = res["price_discount"]
        if has_discount_anchor:
            res["price_discount"] = "нет"

    if has_discount_anchor and res.get("price_discount") in {"", "нет"} and res.get("price_card") not in {"", "нет"}:
        res["price_discount"] = res["price_card"]

    return res

print("[HYP parsers/H2] card/discount anchor disambiguation enabled")
'''

    patches[("parsers", "H4")] = r'''

_prev_normalize_price_candidate = normalize_price_candidate
_prev_extract_price_candidates = extract_price_candidates
_prev_parse_prices = parse_prices

def _price_valid_fmt(v: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d{2}", str(v).strip()))


def _safe_price_float(v: str) -> float | None:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None


def normalize_price_candidate(raw: str) -> str | None:
    s = normalize_whitespace(str(raw))
    s = s.replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    s = re.sub(r"[^\d.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\.\s*", ".", s)
    out = _prev_normalize_price_candidate(s)
    if out and _price_valid_fmt(out):
        return out

    compact = re.sub(r"\s+", "", s)
    if re.fullmatch(r"\d{3,7}", compact):
        out = f"{compact[:-2].lstrip('0') or '0'}.{compact[-2:]}"
        return out if _price_valid_fmt(out) else None
    if re.fullmatch(r"\d{1,5}\.\d{2}", compact):
        return compact
    return None


def _split_decimal_candidates(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    ordered = sorted(tokens, key=lambda t: (t.get("bbox", [0, 0, 0, 0])[1], t.get("bbox", [0, 0, 0, 0])[0]))
    for i, t1 in enumerate(ordered):
        s1 = re.sub(r"\D", "", str(t1.get("text", "")))
        if not re.fullmatch(r"\d{1,4}", s1):
            continue
        b1 = list(t1.get("bbox", [0, 0, 0, 0]))
        if len(b1) != 4:
            continue
        y1c = (b1[1] + b1[3]) / 2.0
        for j in range(i + 1, min(i + 4, len(ordered))):
            t2 = ordered[j]
            s2 = re.sub(r"\D", "", str(t2.get("text", "")))
            if not re.fullmatch(r"\d{2}", s2):
                continue
            b2 = list(t2.get("bbox", [0, 0, 0, 0]))
            if len(b2) != 4:
                continue
            y2c = (b2[1] + b2[3]) / 2.0
            if abs(y2c - y1c) > max(18, 0.40 * max(1, b1[3] - b1[1])):
                continue
            if b2[0] < b1[0]:
                continue
            if (b2[0] - b1[2]) > max(60, 1.5 * max(1, b1[2] - b1[0])):
                continue
            price = f"{int(s1)}.{s2}"
            if not _price_valid_fmt(price):
                continue
            x1 = min(b1[0], b2[0])
            y1 = min(b1[1], b2[1])
            x2 = max(b1[2], b2[2])
            y2 = max(b1[3], b2[3])
            conf = (float(t1.get("confidence", 0.0)) + float(t2.get("confidence", 0.0))) / 2.0
            area = max(1, (x2 - x1) * (y2 - y1))
            out.append(
                {
                    "raw": f"{s1} {s2}",
                    "price": price,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                    "source_text": f"{t1.get('text', '')} {t2.get('text', '')}",
                }
            )
    return out


def extract_price_candidates(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = list(_prev_extract_price_candidates(tokens))
    base.extend(_split_decimal_candidates(tokens))
    if not base:
        return []

    best = {}
    for c in base:
        price = str(c.get("price", "")).strip()
        if not _price_valid_fmt(price):
            continue
        prev = best.get(price)
        score = float(c.get("confidence", 0.0)) * math.log(max(2, int(c.get("area", 1))))
        if prev is None:
            best[price] = c
        else:
            pscore = float(prev.get("confidence", 0.0)) * math.log(max(2, int(prev.get("area", 1))))
            if score > pscore:
                best[price] = c
    return list(best.values())


def parse_prices(tokens: list[dict[str, Any]], color: str, special_symbols: str) -> dict[str, str]:
    result = {"price_default": "", "price_card": "", "price_discount": ""}
    price_cands = extract_price_candidates(tokens)
    if not price_cands:
        return result

    txt = joined_text(tokens).lower()
    has_discount = (color == "promo") or ("discount_percent" in special_symbols) or any(k in txt for k in ["акц", "скид", "выгод", "%"])
    has_card = ("card_price" in special_symbols) or any(k in txt for k in ["карта", "по карте", "card"])

    numeric = []
    for c in price_cands:
        fv = _safe_price_float(c["price"])
        if fv is not None:
            numeric.append((fv, c))
    numeric.sort(key=lambda x: x[0])

    main = sorted(
        price_cands,
        key=lambda c: float(c.get("confidence", 0.0)) * math.log(max(2, int(c.get("area", 1)))),
        reverse=True,
    )[0]

    if len(price_cands) == 1:
        one = str(price_cands[0]["price"])
        if has_discount:
            result["price_discount"] = one
            result["price_default"] = "нет"
            result["price_card"] = "нет"
        else:
            result["price_default"] = one
            result["price_discount"] = "нет"
            result["price_card"] = "нет"
        return result

    min_price = numeric[0][1]["price"] if numeric else main["price"]
    max_price = numeric[-1][1]["price"] if numeric else main["price"]
    mid_price = numeric[1][1]["price"] if len(numeric) >= 3 else min_price

    if has_card and has_discount:
        result["price_card"] = min_price
        result["price_discount"] = mid_price
        result["price_default"] = max_price
    elif has_card:
        result["price_card"] = min_price
        result["price_default"] = max_price
        result["price_discount"] = "нет"
    elif has_discount:
        result["price_discount"] = min_price
        result["price_default"] = max_price
        result["price_card"] = "нет"
    else:
        result["price_default"] = main["price"]
        result["price_card"] = "нет"
        result["price_discount"] = "нет"

    # Strict validity cleanup.
    for fld in ["price_default", "price_card", "price_discount"]:
        v = str(result.get(fld, "")).strip()
        if v and v != "нет" and not _price_valid_fmt(v):
            result[fld] = "нет"

    vd = _safe_price_float(result.get("price_default", ""))
    vp = _safe_price_float(result.get("price_discount", ""))
    if vd is not None and vp is not None and vp > vd:
        result["price_default"], result["price_discount"] = result["price_discount"], result["price_default"]

    return result

print("[HYP parsers/H4] advanced price recovery + strict validity enabled")
'''

    patches[("qr_barcode", "H1")] = r'''

def _ean_checksum_ok(num: str) -> bool:
    if not num.isdigit() or len(num) not in {8, 13}:
        return False
    body, check = num[:-1], int(num[-1])
    if len(num) == 13:
        s = 0
        for i, ch in enumerate(body):
            d = int(ch)
            s += d if i % 2 == 0 else 3 * d
        return ((10 - (s % 10)) % 10) == check
    # EAN-8
    s = 0
    for i, ch in enumerate(body):
        d = int(ch)
        s += 3 * d if i % 2 == 0 else d
    return ((10 - (s % 10)) % 10) == check

_prev_parse_barcode = parse_barcode

def parse_barcode(tokens: list[dict[str, Any]], decoded_barcodes: list[str]) -> str:
    vals = []
    for b in decoded_barcodes:
        digits = re.sub(r"\D", "", str(b))
        if len(digits) in {8, 12, 13, 14}:
            vals.append(digits)
    vals = [v for v in vals if (len(v) in {8, 13} and _ean_checksum_ok(v)) or len(v) in {12, 14}]
    if vals:
        vals.sort(key=lambda x: (len(x), x), reverse=True)
        return vals[0]
    return _prev_parse_barcode(tokens, decoded_barcodes)

print("[HYP qr_barcode/H1] barcode format/checksum validation enabled")
'''

    patches[("track_merge", "H1")] = r'''

def _is_price_valid(v: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d{1,2}", str(v).strip()))


def _best_conf_price(parsed_list: list[dict[str, Any]], field: str) -> str:
    cands = []
    for p in parsed_list:
        v = str(p.get(field, "")).strip()
        if not v or v == "нет":
            continue
        valid = 1 if _is_price_valid(v) else 0
        conf = float(p.get("_ocr_avg_conf", 0.0))
        cands.append((valid, conf, len(v), v))
    if not cands:
        return ""
    cands.sort(reverse=True)
    return cands[0][-1]


_prev_merge_track_results = merge_track_results

def merge_track_results(track_dir: Path, image_results: list[dict[str, Any]]) -> dict[str, Any]:
    for r in image_results:
        q = r.get("quality", {}) if isinstance(r, dict) else {}
        if r.get("parsed"):
            r["parsed"]["_ocr_avg_conf"] = float(q.get("ocr_avg_conf", 0.0))

    out = _prev_merge_track_results(track_dir, image_results)
    parsed_list = [r.get("parsed", {}) for r in image_results if r.get("parsed")]

    row = out.get("row", {})
    for fld in ["price_default", "price_card", "price_discount"]:
        pick = _best_conf_price(parsed_list, fld)
        if pick:
            row[fld] = pick

    bcands = [str(p.get("barcode", "")).strip() for p in parsed_list if str(p.get("barcode", "")).strip()]
    if bcands:
        bcands.sort(key=lambda x: (len(re.sub(r"\D", "", x)), len(x), x), reverse=True)
        row["barcode"] = bcands[0]

    out["row"] = row
    return out

print("[HYP track_merge/H1] confidence-aware merge enabled")
'''

    return patches[key]


def _make_code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip("\n").split("\n")],
    }


def apply_common_rewrites(nb: dict[str, Any]) -> dict[str, Any]:
    cells = nb.get("cells", [])

    # Rewire image collection in track stats cell and main loop cell.
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Disable in-notebook dependency bootstrap cell to avoid PEP668/system pip issues.
        if src.lstrip().startswith("%%bash") and "pip install" in src:
            dep_skip_src = """
import os
print("[DEPS] bootstrap install cell skipped by campaign runner")
print(f"[DEPS] RUN_DEP_INSTALL={os.getenv('RUN_DEP_INSTALL', '0')} (ignored in campaign mode)")
"""
            cell["source"] = [line + "\n" for line in dep_skip_src.rstrip("\n").split("\n")]
            continue
        # Normalize notebook env-check cells that manually purge modules.
        # Some cv2 builds break on forced sys.modules cleanup with circular import.
        if "sys.modules.pop(m, None)" in src and "import numpy, cv2, pandas, matplotlib" in src:
            safe_check_src = """
import numpy, cv2, pandas, matplotlib
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("pandas:", pandas.__version__)
print("matplotlib:", matplotlib.__version__)
"""
            cell["source"] = [line + "\n" for line in safe_check_src.rstrip("\n").split("\n")]
            continue
        src = src.replace(
            'imgs = sorted([p for p in td.iterdir() if p.suffix.lower() in img_exts])',
            'imgs = collect_track_images(td)',
        )
        src = src.replace(
            'image_paths = sorted([p for p in track_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])',
            'image_paths = collect_track_images(track_dir)',
        )
        cell["source"] = [line + "\n" for line in src.rstrip("\n").split("\n")]

    # Insert default collect_track_images before track-stats cell (cell containing TRACK_DIRS assignment).
    track_idx = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if 'TRACK_DIRS = sorted([p for p in DATASET_ROOT.glob("track_*") if p.is_dir()])' in src:
            track_idx = i
            break
    if track_idx is None:
        raise RuntimeError("Could not find TRACK_DIRS cell in notebook")

    default_collector = _make_code_cell(
        """
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_track_images(track_dir: Path) -> list[Path]:
    return sorted([p for p in track_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
"""
    )
    cells.insert(track_idx, default_collector)

    # Re-locate track cell after insertion
    track_idx = track_idx + 1

    # Insert sample/full filtering after track stats cell.
    filter_cell = _make_code_cell(
        """
TRACK_FILTER_FILE = os.getenv("TRACK_FILTER_FILE", "").strip()
VISUAL_PANEL_FILE = os.getenv("VISUAL_PANEL_FILE", "").strip()

if TRACK_FILTER_FILE and Path(TRACK_FILTER_FILE).exists():
    allowed = {
        line.strip()
        for line in Path(TRACK_FILTER_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    TRACK_DIRS = [td for td in TRACK_DIRS if td.name in allowed]

    track_to_images = {}
    all_images = []
    for td in TRACK_DIRS:
        imgs = collect_track_images(td)
        track_to_images[td.name] = imgs
        all_images.extend(imgs)

    images_per_track = [len(v) for v in track_to_images.values()]
    series = pd.Series(images_per_track, name="images_per_track") if images_per_track else pd.Series([], name="images_per_track")
    print(f"[TRACK FILTER] kept tracks: {len(TRACK_DIRS)} / allowed={len(allowed)}")
    print(f"[TRACK FILTER] images total: {len(all_images)}")

if VISUAL_PANEL_FILE and Path(VISUAL_PANEL_FILE).exists():
    panel = [line.strip() for line in Path(VISUAL_PANEL_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
    panel_paths = []
    for tname in panel:
        td = DATASET_ROOT / tname
        if td.exists():
            panel_paths.extend(collect_track_images(td)[:1])
    if panel_paths:
        print(f"[VISUAL PANEL] prepared {len(panel_paths)} panel images")
"""
    )
    cells.insert(track_idx + 1, filter_cell)

    return nb


def apply_hypothesis_cells(nb: dict[str, Any], active_hypotheses: list[tuple[str, str]]) -> dict[str, Any]:
    cells = nb.get("cells", [])

    # Find main processing loop cell.
    loop_idx = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if 'for i, track_dir in enumerate(tqdm(TRACK_DIRS, desc="Processing tracks")):' in src:
            loop_idx = i
            break
    if loop_idx is None:
        raise RuntimeError("Could not find processing loop cell")

    if active_hypotheses:
        hdr = _make_code_cell(
            """
ACTIVE_HYPOTHESES = %s
print("[HYPOTHESES]", ACTIVE_HYPOTHESES)
"""
            % repr(active_hypotheses)
        )
        cells.insert(loop_idx, hdr)
        loop_idx += 1

    for stage, hid in active_hypotheses:
        patch = hypothesis_patch(stage, hid)
        cells.insert(loop_idx, _make_code_cell(patch))
        loop_idx += 1

    return nb


def build_run_notebook(
    src_notebook: Path,
    dst_notebook: Path,
    active_hypotheses: list[tuple[str, str]],
) -> None:
    nb = load_json(src_notebook)
    nb = apply_common_rewrites(nb)
    nb = apply_hypothesis_cells(nb, active_hypotheses)
    dump_json(dst_notebook, nb)


def execute_notebook(
    notebook_path: Path,
    executed_path: Path,
    env: dict[str, str],
    timeout: int,
    jupyter_cmd: str,
) -> None:
    cmd = [
        jupyter_cmd,
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        str(executed_path),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    started = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - started
    print(f"[EXEC] notebook done in {elapsed:.1f}s")


def default_state() -> dict[str, Any]:
    return {
        "champion_run_id": None,
        "accepted_hypotheses": [],
        "runs": {},
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    return load_json(path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    dump_json(path, state)


def _run_sort_key(run_id: str, run_info: dict[str, Any]) -> tuple[str, str]:
    meta = run_info.get("meta", {}) if isinstance(run_info, dict) else {}
    created = str(meta.get("created_at_utc", ""))
    return (created, run_id)


def write_candidates_table(output_root: Path, state: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs = state.get("runs", {}) if isinstance(state.get("runs", {}), dict) else {}
    ordered = sorted(runs.items(), key=lambda kv: _run_sort_key(kv[0], kv[1]))

    for run_id, info in ordered:
        meta = info.get("meta", {}) if isinstance(info, dict) else {}
        metrics_raw = info.get("metrics", {}) if isinstance(info, dict) else {}
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        row = {
            "run_id": run_id,
            "status": str(meta.get("status", "")),
            "mode": str(meta.get("mode", "")),
            "active_hypotheses": ";".join(meta.get("active_hypotheses", []) or []),
            "rows": metrics.get("rows", ""),
            "proxy_score": f"{float(metrics.get('proxy_score', 0.0)):.5f}" if metrics else "",
            "case_proxy_v2": f"{float(metrics.get('case_proxy_v2', 0.0)):.5f}" if metrics else "",
            "official_final_score": "" if metrics.get("official_final_score") is None else f"{float(metrics.get('official_final_score', 0.0)):.5f}",
            "ranking_mode": str(metrics.get("ranking_mode", "")),
            "ranking_score": f"{float(metrics.get('ranking_score', 0.0)):.5f}" if metrics else "",
            "elapsed_sec": f"{float(metrics.get('elapsed_sec', 0.0)):.2f}" if metrics else "",
        }
        rows.append(row)

    csv_path = output_root / "candidate_table.csv"
    md_path = output_root / "candidate_table.md"

    if rows:
        headers = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
        headers = [
            "run_id",
            "status",
            "mode",
            "active_hypotheses",
            "rows",
            "proxy_score",
            "case_proxy_v2",
            "official_final_score",
            "ranking_mode",
            "ranking_score",
            "elapsed_sec",
        ]

    md_lines = [
        "# Candidate Comparison",
        "",
        as_markdown_row(headers),
        as_markdown_row(["---"] * len(headers)),
    ]
    for r in rows:
        md_lines.append(as_markdown_row([str(r.get(h, "")) for h in headers]))
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def make_run_dirs(output_root: Path, run_id: str) -> dict[str, Path]:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "nb_path": run_dir / "candidate.ipynb",
        "exec_nb_path": run_dir / "candidate.executed.ipynb",
        "result_dir": run_dir / "outputs_ocr_baseline",
        "result_csv": run_dir / "outputs_ocr_baseline" / "result.csv",
        "meta_path": run_dir / "run_meta.json",
        "metrics_path": run_dir / "proxy_metrics.json",
        "metrics_v2_path": run_dir / "metrics_v2.json",
        "official_report_path": run_dir / "official_eval_report.json",
        "official_matches_path": run_dir / "official_eval_matches.csv",
        "track_filter": run_dir / "track_filter.txt",
        "visual_panel": run_dir / "visual_panel.txt",
    }


def resolve_evaluate_script(project_root: Path, explicit_path: str) -> Path | None:
    if explicit_path.strip():
        p = Path(explicit_path).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if p.exists():
            return p
        return None
    candidates = [
        project_root / "evaluate_matching.py",
        project_root / "Parser" / "evaluate_matching.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def try_evaluate_official(
    *,
    project_root: Path,
    pred_csv: Path,
    gt_csv: Path | None,
    out_report: Path,
    out_matches: Path,
    evaluate_script: str,
    python_cmd: str,
    time_tolerance_ms: float,
    iou_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "mode": "not_requested",
    }
    if gt_csv is None:
        return result
    if not str(gt_csv).strip():
        return result
    if not gt_csv.exists():
        return {
            "available": False,
            "mode": "gt_missing",
            "error": f"GT CSV not found: {gt_csv}",
        }

    script_path = resolve_evaluate_script(project_root, evaluate_script)
    if script_path is None:
        return {
            "available": False,
            "mode": "script_missing",
            "error": "evaluate_matching.py not found",
        }

    cmd = [
        python_cmd or sys.executable,
        str(script_path),
        "--pred",
        str(pred_csv),
        "--gt",
        str(gt_csv),
        "--out",
        str(out_report),
        "--matches-out",
        str(out_matches),
        "--time-tolerance-ms",
        str(time_tolerance_ms),
        "--iou-threshold",
        str(iou_threshold),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        return {
            "available": False,
            "mode": "execution_failed",
            "error": str(exc),
            "script_path": str(script_path),
        }

    if not out_report.exists():
        return {
            "available": False,
            "mode": "report_missing",
            "error": f"Expected report not found: {out_report}",
            "script_path": str(script_path),
        }

    try:
        report = load_json(out_report)
    except Exception as exc:
        return {
            "available": False,
            "mode": "report_parse_failed",
            "error": str(exc),
            "script_path": str(script_path),
            "report_path": str(out_report),
        }

    return {
        "available": True,
        "mode": "ok",
        "script_path": str(script_path),
        "report_path": str(out_report),
        "matches_path": str(out_matches),
        "report": report,
    }


def run_once(
    *,
    run_id: str,
    mode: str,
    project_root: Path,
    dataset_root: Path,
    task_path: Path,
    output_root: Path,
    notebook_name: str,
    active_hypotheses: list[tuple[str, str]],
    sample_size: int,
    visual_panel_size: int,
    seed: int,
    timeout: int,
    jupyter_cmd: str,
    dry_run: bool,
    gt_csv: Path | None,
    evaluate_script: str,
    python_cmd: str,
    eval_time_tolerance_ms: float,
    eval_iou_threshold: float,
    products_dict_csv: Path | None,
    google_dict_csv: Path | None,
) -> dict[str, Any]:
    paths = make_run_dirs(output_root, run_id)

    all_tracks = discover_tracks(dataset_root)
    if not all_tracks:
        raise RuntimeError(f"No track_* directories found under {dataset_root}")

    if mode == "sample":
        sample_names = pick_sample_tracks(all_tracks, sample_size=sample_size, seed=seed)
    elif mode == "full":
        sample_names = [p.name for p in all_tracks]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    panel_names = pick_visual_panel(sample_names, visual_size=visual_panel_size, seed=seed)

    paths["track_filter"].write_text("\n".join(sample_names) + "\n", encoding="utf-8")
    paths["visual_panel"].write_text("\n".join(panel_names) + "\n", encoding="utf-8")

    src_notebook = project_root / notebook_name
    if not src_notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {src_notebook}")

    build_run_notebook(src_notebook, paths["nb_path"], active_hypotheses)

    meta: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "task_path": str(task_path),
        "active_hypotheses": [f"{s}/{h}" for s, h in active_hypotheses],
        "tracks_selected": sample_names,
        "visual_panel_tracks": panel_names,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "elapsed_sec": None,
        "gt_csv": str(gt_csv) if gt_csv else "",
        "evaluate_script": evaluate_script,
    }

    if dry_run:
        meta["status"] = "dry_run"
        dump_json(paths["meta_path"], meta)
        return {
            "meta": meta,
            "paths": {k: str(v) for k, v in paths.items()},
            "metrics": None,
        }

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project_root),
            "DATASET_ROOT": str(dataset_root),
            "TASK_PATH": str(task_path),
            "OUTPUT_DIR": str(paths["result_dir"]),
            "OUTPUT_CSV": str(paths["result_csv"]),
            "TRACK_FILTER_FILE": str(paths["track_filter"]),
            "VISUAL_PANEL_FILE": str(paths["visual_panel"]),
            "MPLBACKEND": "Agg",
            "PADDLE_PDX_EAGER_INIT": "False",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        }
    )
    if products_dict_csv is not None:
        env["PRODUCTS_DICT_CSV"] = str(products_dict_csv)
    if google_dict_csv is not None:
        env["GOOGLE_DICT_CSV"] = str(google_dict_csv)

    started = time.time()
    execute_notebook(paths["nb_path"], paths["exec_nb_path"], env=env, timeout=timeout, jupyter_cmd=jupyter_cmd)
    elapsed = time.time() - started

    if not paths["result_csv"].exists():
        raise RuntimeError(f"Run finished but result CSV not found: {paths['result_csv']}")

    metrics = compute_proxy_metrics(paths["result_csv"])
    metrics["elapsed_sec"] = elapsed
    official = try_evaluate_official(
        project_root=project_root,
        pred_csv=paths["result_csv"],
        gt_csv=gt_csv,
        out_report=paths["official_report_path"],
        out_matches=paths["official_matches_path"],
        evaluate_script=evaluate_script,
        python_cmd=python_cmd,
        time_tolerance_ms=eval_time_tolerance_ms,
        iou_threshold=eval_iou_threshold,
    )
    metrics["official"] = official
    if official.get("available"):
        report = official.get("report", {}) if isinstance(official.get("report"), dict) else {}
        metrics["official_final_score"] = float(report.get("final_score", 0.0))
        metrics["ranking_mode"] = "official_final_score"
        metrics["ranking_score"] = float(metrics["official_final_score"])
    else:
        metrics["official_final_score"] = None
        metrics["ranking_mode"] = "case_proxy_v2"
        metrics["ranking_score"] = float(metrics.get("case_proxy_v2", metrics.get("proxy_score", 0.0)))

    meta["status"] = "completed"
    meta["elapsed_sec"] = elapsed

    dump_json(paths["meta_path"], meta)
    dump_json(paths["metrics_path"], metrics)
    dump_json(paths["metrics_v2_path"], metrics)

    return {
        "meta": meta,
        "paths": {k: str(v) for k, v in paths.items()},
        "metrics": metrics,
    }


def _parse_hypothesis(hyp_text: str) -> tuple[str, str]:
    if "/" in hyp_text:
        stage, hid = hyp_text.split("/", 1)
    elif ":" in hyp_text:
        stage, hid = hyp_text.split(":", 1)
    else:
        raise ValueError("Hypothesis must be in 'stage/Hx' format")
    stage = stage.strip()
    hid = hid.strip()
    if (stage, hid) not in HYP_INDEX:
        raise ValueError(f"Unknown hypothesis: {stage}/{hid}")
    return stage, hid


def _parse_hypothesis_tags(tags: list[str]) -> list[tuple[str, str]]:
    return [_parse_hypothesis(t) for t in tags]


def _resolve_optional_path(project_root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def _common_run_options(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    return {
        "gt_csv": _resolve_optional_path(project_root, getattr(args, "gt_csv", "")),
        "evaluate_script": str(getattr(args, "evaluate_script", "") or ""),
        "python_cmd": str(getattr(args, "python_cmd", "python3") or "python3"),
        "eval_time_tolerance_ms": float(getattr(args, "eval_time_tolerance_ms", 500.0)),
        "eval_iou_threshold": float(getattr(args, "eval_iou_threshold", 0.3)),
        "products_dict_csv": _resolve_optional_path(project_root, getattr(args, "products_dict_csv", "")),
        "google_dict_csv": _resolve_optional_path(project_root, getattr(args, "google_dict_csv", "")),
    }


def _sanitize_id(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "bundle"


def _parse_bundle_spec(spec: str, idx: int = 1) -> tuple[str, list[str], list[tuple[str, str]]]:
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty bundle specification")

    name = ""
    body = raw
    if "=" in raw:
        maybe_name, maybe_body = raw.split("=", 1)
        if "/" in maybe_body:
            name = maybe_name.strip()
            body = maybe_body
    elif ":" in raw:
        maybe_name, maybe_body = raw.split(":", 1)
        if "/" in maybe_body:
            name = maybe_name.strip()
            body = maybe_body

    tags = [t.strip() for t in body.split(",") if t.strip()]
    if not tags:
        raise ValueError(f"Invalid bundle specification: {spec}")
    pairs = _parse_hypothesis_tags(tags)
    tags_norm = [f"{s}/{h}" for s, h in pairs]
    if not name:
        name = f"bundle_{idx:02d}"
    return name, tags_norm, pairs


def cmd_run_bundle(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    task_path = Path(args.task_path).resolve()
    output_root = (project_root / args.output_root).resolve()
    state_path = output_root / "campaign_state.json"
    common_opts = _common_run_options(args, project_root)

    raw_bundle = (args.bundle or "").strip()
    if raw_bundle:
        _, requested_tags, _ = _parse_bundle_spec(raw_bundle, idx=1)
    else:
        requested_tags = list(CHAMPION_BUNDLE_TAGS)

    # Final package is intentionally locked to one champion configuration.
    if set(requested_tags) != set(CHAMPION_BUNDLE_TAGS):
        raise ValueError(
            "Only champion bundle is allowed in final package. "
            f"Expected tags: {CHAMPION_BUNDLE_TAGS}"
        )

    active_tags = list(CHAMPION_BUNDLE_TAGS)
    active_pairs = _parse_hypothesis_tags(active_tags)
    run_id = f"bundle_{_sanitize_id(CHAMPION_BUNDLE_NAME)}_{args.mode}_{now_utc()}"
    result = run_once(
        run_id=run_id,
        mode=args.mode,
        project_root=project_root,
        dataset_root=dataset_root,
        task_path=task_path,
        output_root=output_root,
        notebook_name=args.notebook,
        active_hypotheses=active_pairs,
        sample_size=args.sample_size,
        visual_panel_size=args.visual_panel_size,
        seed=args.seed,
        timeout=args.timeout,
        jupyter_cmd=args.jupyter_cmd,
        dry_run=args.dry_run,
        **common_opts,
    )

    result.setdefault("meta", {})["bundle_name"] = CHAMPION_BUNDLE_NAME
    result["meta"]["bundle_tags"] = list(CHAMPION_BUNDLE_TAGS)
    result["meta"]["active_hypotheses"] = active_tags

    state = load_state(state_path)
    state.setdefault("runs", {})[run_id] = result
    state["champion_run_id"] = run_id
    state["accepted_hypotheses"] = list(CHAMPION_BUNDLE_TAGS)

    save_state(state_path, state)
    write_candidates_table(output_root, state)

    print(f"[BUNDLE] run_id={run_id}")
    print(f"[BUNDLE] name={CHAMPION_BUNDLE_NAME}")
    print(f"[BUNDLE] tags={CHAMPION_BUNDLE_TAGS}")
    if args.dry_run:
        print("[BUNDLE] dry-run prepared")
        return
    metrics = result.get("metrics") or {}
    print(
        "[BUNDLE] "
        f"proxy_score={float(metrics.get('proxy_score', 0.0)):.5f} "
        f"case_proxy_v2={float(metrics.get('case_proxy_v2', 0.0)):.5f} "
        f"ranking={metrics.get('ranking_mode')}:{float(metrics.get('ranking_score', 0.0)):.5f}"
    )


def cmd_show_state(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    state_path = output_root / "campaign_state.json"
    state = load_state(state_path)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final OCR runner (champion bundle only)")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=".")
    common.add_argument("--dataset-root", default="./top_crops")
    common.add_argument("--task-path", default="./lenta_tech_life_hack_text.md")
    common.add_argument("--notebook", default=DEFAULT_NOTEBOOK)
    common.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    common.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    common.add_argument("--visual-panel-size", type=int, default=DEFAULT_VISUAL_PANEL_SIZE)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--timeout", type=int, default=-1)
    common.add_argument("--jupyter-cmd", default="jupyter")
    common.add_argument("--gt-csv", default="", help="Optional GT CSV path. If provided, ranking switches to official final_score.")
    common.add_argument("--evaluate-script", default="", help="Optional path to evaluate_matching.py. Auto-detected if empty.")
    common.add_argument("--python-cmd", default="python3", help="Python executable for official evaluation.")
    common.add_argument("--eval-time-tolerance-ms", type=float, default=500.0)
    common.add_argument("--eval-iou-threshold", type=float, default=0.3)
    common.add_argument("--products-dict-csv", default="./products_v2_merged.csv", help="Catalog CSV used by name-correction hypotheses.")
    common.add_argument("--google-dict-csv", default="", help="Optional second dictionary CSV (Google dataset).")
    common.add_argument("--dry-run", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run_bundle", parents=[common])
    sp.add_argument("--mode", choices=["sample", "full"], default="sample")
    sp.add_argument(
        "--bundle",
        default=f"{CHAMPION_BUNDLE_NAME}={','.join(CHAMPION_BUNDLE_TAGS)}",
        help="Ignored unless it is exactly champion bundle tags.",
    )
    sp.set_defaults(func=cmd_run_bundle)

    sp = sub.add_parser("show_state", parents=[common])
    sp.set_defaults(func=cmd_show_state)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
