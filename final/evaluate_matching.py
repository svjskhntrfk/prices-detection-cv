#!/usr/bin/env python3
"""
Evaluate price tag detection results against ground truth CSV.
Usage:
  python evaluate_matching.py --pred result.csv --gt ground_truth.csv \
    --out report.json --matches-out matches.csv \
    --time-tolerance-ms 500 --iou-threshold 0.3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Fields used only for matching — not evaluated as content
MATCH_KEY_FIELDS = {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}

# Numeric price fields
PRICE_FIELDS = {
    "price_default", "price_card", "price_discount",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "action_price_qr",
    "wholesale_level_1_price", "wholesale_level_2_price",
}

# "No value" sentinel strings
NONE_VALUES = {"нет", "none", "null", "nan", ""}

ROW_ACCURACY_THRESHOLD = 0.8


def normalize(value, field: str) -> str | float | None:
    """Return normalised comparable value, or None meaning 'absent'.

    Normalization pipeline by field type:

    ALL FIELDS
      NaN / None                     → None   (pandas missing)
      strip + lower                  → e.g. "  Нет  " → "нет"
      "нет" | "none" | "null"
        | "nan" | ""                 → None   (treated as absent)

    PRICE_FIELDS  (price_default, price_card, price_discount,
                   price1_qr..price4_qr, action_price_qr,
                   wholesale_level_1_price, wholesale_level_2_price)
      comma → dot, spaces removed    → "3 789,49" → "3789.49"
      parsed as float                → 3789.49
      unparseable (text remnant)     → kept as lowercase str

    CODE FIELDS  (barcode, id_sku, code, qr_code_barcode, action_code_qr)
      strip non-word chars [^\w]     → "350 061-011.7022" → "3500610117022"
      result is lowercase str        → "ABC123" → "abc123"

    OTHER TEXT FIELDS  (product_name, color, additional_info, etc.)
      only strip + lower applied     → "Сухое" → "сухое"
    """
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in NONE_VALUES:
        return None

    if field in PRICE_FIELDS:
        s = s.replace(",", ".").replace(" ", "")
        try:
            return float(s)
        except ValueError:
            return s  # keep as string if unparseable

    # barcode / id_sku / code — strip non-alnum
    if field in {"barcode", "id_sku", "code", "qr_code_barcode", "action_code_qr"}:
        return re.sub(r"[^\w]", "", s)

    return s


def iou(r1, r2) -> float:
    """Compute IoU for two bounding boxes (x_min, y_min, x_max, y_max)."""
    xi1 = max(r1["x_min"], r2["x_min"])
    yi1 = max(r1["y_min"], r2["y_min"])
    xi2 = min(r1["x_max"], r2["x_max"])
    yi2 = min(r1["y_max"], r2["y_max"])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    a1 = (r1["x_max"] - r1["x_min"]) * (r1["y_max"] - r1["y_min"])
    a2 = (r2["x_max"] - r2["x_min"]) * (r2["y_max"] - r2["y_min"])
    return inter / (a1 + a2 - inter)


def to_float(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", "."))
    except (ValueError, TypeError):
        return default


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ("x_min", "y_min", "x_max", "y_max"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: to_float(v))
    if "frame_timestamp" in df.columns:
        df["frame_timestamp"] = df["frame_timestamp"].apply(lambda v: to_float(v))
    return df


def barcode_key(row) -> str | None:
    """Return normalised barcode or None if absent."""
    if "barcode" not in row.index:
        return None
    v = normalize(row["barcode"], "barcode")
    return v if v else None


def content_fields(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in MATCH_KEY_FIELDS]


def row_accuracy(pred_row, gt_row, fields: list[str]) -> tuple[float, list[str]]:
    """Return (accuracy 0-1, list of wrong field names)."""
    evaluated, correct = 0, 0
    errors = []
    for f in fields:
        gt_val = normalize(gt_row.get(f), f) if f in gt_row.index else None
        pred_val = normalize(pred_row.get(f), f) if f in pred_row.index else None

        if gt_val is None:
            # GT has no value — only error if pred fabricates something meaningful
            # (we choose not to penalise extra info from pred here)
            continue
        evaluated += 1
        if pred_val == gt_val:
            correct += 1
        else:
            errors.append(f)

    if evaluated == 0:
        return 1.0, []
    return correct / evaluated, errors


def match(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
          time_tol_ms: float, iou_thresh: float):

    # Build barcode index for GT
    gt_barcode_index: dict[str, int] = {}
    for idx, row in gt_df.iterrows():
        bc = barcode_key(row)
        if bc:
            gt_barcode_index[bc] = idx

    gt_used: set[int] = set()
    records = []  # (pred_idx, gt_idx|None, match_type, score)

    # --- Pass 1: barcode matching ---
    unmatched_pred = []
    for pidx, prow in pred_df.iterrows():
        bc = barcode_key(prow)
        if bc and bc in gt_barcode_index:
            gidx = gt_barcode_index[bc]
            if gidx not in gt_used:
                gt_used.add(gidx)
                records.append((pidx, gidx, "barcode", 1.0))
                continue
        unmatched_pred.append(pidx)

    # --- Pass 2: spatial-temporal matching ---
    has_bbox = all(c in pred_df.columns for c in ("x_min", "y_min", "x_max", "y_max"))
    has_ts = "frame_timestamp" in pred_df.columns

    still_unmatched = []
    for pidx in unmatched_pred:
        prow = pred_df.loc[pidx]
        p_fn = str(prow.get("filename", "")).strip() if "filename" in prow.index else None
        p_ts = float(prow["frame_timestamp"]) if has_ts else None
        p_box = {k: float(prow[k]) for k in ("x_min", "y_min", "x_max", "y_max")} if has_bbox else None

        best_gidx, best_score = None, -1.0
        for gidx, grow in gt_df.iterrows():
            if gidx in gt_used:
                continue
            # filename must match if present
            if p_fn is not None and "filename" in grow.index:
                if str(grow["filename"]).strip() != p_fn:
                    continue

            # timestamp check
            ts_err = 0.0
            if has_ts and p_ts is not None and "frame_timestamp" in grow.index:
                g_ts = float(grow["frame_timestamp"])
                dt = abs(p_ts - g_ts)
                if dt > time_tol_ms:
                    continue
                ts_err = dt / max(time_tol_ms, 1)

            # bbox IoU check
            iou_val = 0.0
            if has_bbox and p_box is not None:
                g_box = {k: float(grow[k]) for k in ("x_min", "y_min", "x_max", "y_max")
                         if k in grow.index}
                if len(g_box) == 4:
                    iou_val = iou(p_box, g_box)
                    if iou_val < iou_thresh:
                        continue

            score = iou_val - 0.1 * ts_err
            if score > best_score:
                best_score = score
                best_gidx = gidx

        if best_gidx is not None:
            gt_used.add(best_gidx)
            records.append((pidx, best_gidx, "spatial", best_score))
        else:
            still_unmatched.append(pidx)

    for pidx in still_unmatched:
        records.append((pidx, None, "unmatched", 0.0))

    return records, gt_used


def evaluate(pred_path: str, gt_path: str, out_path: str, matches_out: str,
             time_tol_ms: float, iou_thresh: float):

    pred_df = load_csv(pred_path)
    gt_df = load_csv(gt_path)

    records, gt_matched_idxs = match(pred_df, gt_df, time_tol_ms, iou_thresh)

    fields = content_fields(gt_df)
    # Make sure pred has same fields (use union)
    all_fields = list(dict.fromkeys(fields + content_fields(pred_df)))

    match_rows = []
    matched_by_barcode = 0
    matched_by_spatial = 0
    field_correct: dict[str, int] = {f: 0 for f in all_fields}
    field_total: dict[str, int] = {f: 0 for f in all_fields}
    high_accuracy_gt: set = set()

    for pidx, gidx, mtype, mscore in records:
        prow = pred_df.loc[pidx]
        if gidx is not None:
            grow = gt_df.loc[gidx]
            acc, errors = row_accuracy(prow, grow, all_fields)
            # per-field stats
            for f in all_fields:
                gt_val = normalize(grow.get(f), f) if f in grow.index else None
                if gt_val is None:
                    continue
                field_total[f] = field_total.get(f, 0) + 1
                pred_val = normalize(prow.get(f), f) if f in prow.index else None
                if pred_val == gt_val:
                    field_correct[f] = field_correct.get(f, 0) + 1
            if mtype == "barcode":
                matched_by_barcode += 1
            else:
                matched_by_spatial += 1
            if acc >= ROW_ACCURACY_THRESHOLD:
                high_accuracy_gt.add(gidx)
        else:
            acc, errors = 0.0, []

        match_rows.append({
            "pred_index": pidx,
            "gt_index": gidx if gidx is not None else "",
            "match_type": mtype,
            "match_score": round(mscore, 4),
            "row_accuracy": round(acc, 4),
            "field_errors": "|".join(errors),
        })

    total_gt = len(gt_df)
    total_pred = len(pred_df)
    unmatched_result = sum(1 for _, gidx, _, _ in records if gidx is None)
    unmatched_gt = total_gt - len(gt_matched_idxs)
    final_score = len(high_accuracy_gt) / total_gt if total_gt else 0.0

    field_accuracy = {
        f: round(field_correct[f] / field_total[f], 4) if field_total[f] else None
        for f in all_fields
    }

    # ------------------------------------------------------------------
    # report.json — aggregated metrics
    # ------------------------------------------------------------------
    # total_gt              : total rows in ground-truth CSV
    # total_pred            : total rows in prediction CSV
    # matched_by_barcode    : pred rows matched via barcode (pass 1)
    # matched_by_spatial    : pred rows matched via IoU + timestamp (pass 2)
    # unmatched_result      : pred rows that could not be matched to any GT row
    #                         (neither barcode nor spatial) — counted as missed detections
    # unmatched_gt          : GT rows that no pred row was matched to
    #                         (= total_gt - len(gt_matched_idxs))
    # final_score           : main task metric:
    #                           count(GT rows with matched pred AND row_accuracy >= 0.8)
    #                           ─────────────────────────────────────────────────────────
    #                                          total_gt
    #                         Range [0, 1]. 1.0 = every GT ценник found and ≥80% fields correct.
    # row_accuracy_threshold: the 0.8 cutoff used in final_score (constant, written for traceability)
    # time_tolerance_ms     : --time-tolerance-ms value used in this run
    # iou_threshold         : --iou-threshold value used in this run
    # field_accuracy        : dict  field_name → float [0, 1]
    #                         Computed only over matched pairs where GT has a non-None value.
    #                         Fields where GT is always None are excluded (no denominator).
    #                         Formula per field:
    #                           correct_matches / total_gt_rows_where_field_is_present
    # ------------------------------------------------------------------
    report = {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "matched_by_barcode": matched_by_barcode,
        "matched_by_spatial": matched_by_spatial,
        "unmatched_result": unmatched_result,
        "unmatched_gt": unmatched_gt,
        "final_score": round(final_score, 4),
        "row_accuracy_threshold": ROW_ACCURACY_THRESHOLD,
        "time_tolerance_ms": time_tol_ms,
        "iou_threshold": iou_thresh,
        "field_accuracy": {f: v for f, v in field_accuracy.items() if v is not None},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # matches.csv — one row per pred row
    # ------------------------------------------------------------------
    # pred_index   : integer index of the row in the prediction CSV (0-based)
    # gt_index     : integer index of the matched GT row, or "" if unmatched
    # match_type   : "barcode"   — matched in pass 1 by barcode equality
    #                "spatial"   — matched in pass 2 by filename + IoU + timestamp
    #                "unmatched" — no GT row found; row not counted in final_score
    # match_score  : quality of the spatial match (float):
    #                  barcode   → always 1.0
    #                  spatial   → iou - 0.1 * (|Δt| / time_tolerance_ms)
    #                              range ≈ [iou_threshold - 0.1, 1.0]
    #                  unmatched → 0.0
    # row_accuracy : fraction of content fields that matched GT after normalization
    #                  = correct_fields / fields_where_gt_is_not_None
    #                  unmatched rows → 0.0
    #                  rows where GT has no non-None fields → 1.0 (nothing to check)
    # field_errors : pipe-separated list of field names where pred ≠ gt (after normalization)
    #                e.g. "price_default|id_sku"
    #                empty string means all evaluated fields matched
    # ------------------------------------------------------------------
    matches_df = pd.DataFrame(match_rows)
    matches_df.to_csv(matches_out, index=False, encoding="utf-8-sig")

    # Console summary
    print(f"\n=== Evaluation Summary ===")
    print(f"  GT rows:            {total_gt}")
    print(f"  Pred rows:          {total_pred}")
    print(f"  Matched by barcode: {matched_by_barcode}")
    print(f"  Matched by spatial: {matched_by_spatial}")
    print(f"  Unmatched pred:     {unmatched_result}")
    print(f"  Unmatched GT:       {unmatched_gt}")
    print(f"  High-accuracy GT:   {len(high_accuracy_gt)} (acc >= {ROW_ACCURACY_THRESHOLD})")
    print(f"  FINAL SCORE:        {final_score:.4f}")
    print(f"\n  Field accuracy (top fields):")
    for f, acc in sorted(field_accuracy.items(), key=lambda x: -(x[1] or 0))[:10]:
        if acc is not None:
            print(f"    {f:<35} {acc:.2%}")
    print(f"\n  Report  -> {out_path}")
    print(f"  Matches -> {matches_out}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate price-tag detection results.")
    parser.add_argument("--pred", required=True, help="Path to prediction CSV")
    parser.add_argument("--gt", required=True, help="Path to ground-truth CSV")
    parser.add_argument("--out", default="report.json", help="Output report JSON")
    parser.add_argument("--matches-out", default="matches.csv", help="Output matches CSV")
    parser.add_argument("--time-tolerance-ms", type=float, default=500,
                        help="Max timestamp difference in ms for spatial matching")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="Minimum IoU for spatial matching")
    args = parser.parse_args()

    for p in (args.pred, args.gt):
        if not Path(p).exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    evaluate(
        pred_path=args.pred,
        gt_path=args.gt,
        out_path=args.out,
        matches_out=args.matches_out,
        time_tol_ms=args.time_tolerance_ms,
        iou_thresh=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
