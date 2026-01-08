from __future__ import annotations

import os
import re
import json
import time
import zipfile
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# Dependencies:
#   pip install nibabel pillow numpy scipy
import nibabel as nib
from PIL import Image

# Try import scipy (recommended). Fallback if not available.
try:
    from scipy import ndimage as ndi
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# Config
# -----------------------------
OUT_SIZE = (288, 288)
DEFAULT_KEEP = (0.05, 0.95)        # keep middle slices to reduce empty ends
DEFAULT_MIN_CONTENT = 0.01         # filter extremely empty slices
DEFAULT_AXIS = 2                   # axial after canonical


# -----------------------------
# Logging
# -----------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# -----------------------------
# Basic utilities
# -----------------------------
def robust_norm_01(vol: np.ndarray, pmin: float = 1, pmax: float = 99, eps: float = 1e-8) -> np.ndarray:
    """Percentile normalize -> [0,1]."""
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, [pmin, pmax])
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + eps)
    return v


def get_subject_id(path: Path) -> str:
    """Extract subject id from filename."""
    name = path.name
    m = re.search(r"(CC\d+|sub-\d+)", name)
    if m:
        return m.group(1)
    return path.stem.replace(".nii", "")


def find_any_nifti(root: Path) -> List[Path]:
    return sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))


def already_extracted(extract_dir: Path) -> bool:
    if not extract_dir.exists():
        return False
    return len(find_any_nifti(extract_dir)) > 0


def safe_unzip(zip_path: Path, extract_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting: {zip_path} -> {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # zip-slip protection
        base = extract_dir.resolve()
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError("Unsafe zip content detected (zip slip).")
        zf.extractall(extract_dir)

    log("Extraction complete.")


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            log("Warning: manifest exists but failed to read. Recreating.")
    return {"created_at": time.time(), "subjects": {}}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def should_skip(nifti_path: Path, out_path: Path, manifest: Dict[str, Any]) -> bool:
    sid = get_subject_id(nifti_path)
    rec = manifest.get("subjects", {}).get(sid)
    if not out_path.exists() or not rec:
        return False
    try:
        return float(rec.get("mtime_in", -1)) == float(os.path.getmtime(nifti_path))
    except Exception:
        return False


# -----------------------------
# Resize + debug save
# -----------------------------
def resize2d_01(x01: np.ndarray, size: Tuple[int, int] = OUT_SIZE) -> np.ndarray:
    """
    Resize float32 [0,1] slice using 16-bit intermediate (preserve contrast),
    then back to float32 [0,1].
    """
    x16 = (np.clip(x01, 0.0, 1.0) * 65535.0).astype(np.uint16)
    im = Image.fromarray(x16, mode="I;16")
    im = im.resize(size, Image.BILINEAR)
    y16 = np.array(im, dtype=np.uint16)
    return y16.astype(np.float32) / 65535.0


def save_jpg_from_01(img01: np.ndarray, out_path: Path, quality: int = 95) -> None:
    """Save float [0,1] image as JPG (8-bit) for visual recheck only."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr8, mode="L")
    im.save(str(out_path), format="JPEG", quality=int(quality), optimize=True)


def black_ratio(img01: np.ndarray, black_thresh: float = 0.02) -> float:
    """
    img01: float [0,1]
    pixel <= black_thresh considered black
    """
    x = np.asarray(img01, dtype=np.float32)
    return float((x <= black_thresh).mean())

def tight_intensity_crop(img01: np.ndarray, black_thresh: float = 0.02, margin: int = 2, min_area_ratio: float = 0.005):
    """
    Crop ให้ชิดบริเวณที่ไม่ดำ (img > black_thresh)
    - margin: เผื่อขอบนิดหน่อย กันตัดเนื้อหัว
    - min_area_ratio: ถ้าบริเวณไม่ดำเล็กเกินไป จะไม่ crop (กันเพี้ยน)
    """
    x = np.asarray(img01, dtype=np.float32)
    h, w = x.shape
    mask = x > float(black_thresh)

    if float(mask.mean()) < float(min_area_ratio):
        return x  # ไม่ crop ถ้า foreground น้อยเกินไป

    ys, xs = np.where(mask)
    if ys.size == 0:
        return x

    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1

    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(h, r1 + margin)
    c1 = min(w, c1 + margin)

    # กัน crop จนเล็กเกินไป
    if (r1 - r0) < 16 or (c1 - c0) < 16:
        return x

    return x[r0:r1, c0:c1]



# -----------------------------
# Head (skull+brain) crop helpers
# -----------------------------
def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold for 1D values."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return vmin
    hist, bin_edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return float(centers[idx])


def head_mask_and_crop(
    img01: np.ndarray,
    margin: int = 12,
    dilate_iter: int = 2,
    close_iter: int = 2,
    min_area_ratio: float = 0.01,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop head region (skull+brain) from a 2D slice:
    - threshold (Otsu on non-zero region)
    - closing + fill holes + largest component (requires scipy)
    - dilation to include skull edge
    Returns cropped image and bbox (r0,r1,c0,c1) on original.
    """
    x = img01.astype(np.float32)
    h, w = x.shape

    # fallback if scipy not available
    if not SCIPY_OK:
        fg = x[x > 0]
        if fg.size < 100:
            return x, (0, h, 0, w)
        thr = np.percentile(fg, 20)
        mask = x > thr
        ys, xs = np.where(mask)
        if ys.size == 0:
            return x, (0, h, 0, w)
        r0, r1 = ys.min(), ys.max() + 1
        c0, c1 = xs.min(), xs.max() + 1
        r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
        r1 = min(h, r1 + margin); c1 = min(w, c1 + margin)
        return x[r0:r1, c0:c1], (r0, r1, c0, c1)

    fg = x[x > 0]
    if fg.size < 100:
        return x, (0, h, 0, w)

    thr = otsu_threshold(fg, nbins=256)
    mask = x > thr

    if close_iter > 0:
        mask = ndi.binary_closing(mask, iterations=close_iter)
    mask = ndi.binary_fill_holes(mask)

    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
        best = int(np.argmax(sizes)) + 1
        mask = lab == best

    if mask.mean() < min_area_ratio:
        mask = x > 0
        mask = ndi.binary_fill_holes(mask)
        lab, n = ndi.label(mask)
        if n > 1:
            sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
            best = int(np.argmax(sizes)) + 1
            mask = lab == best

    if dilate_iter > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate_iter)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return x, (0, h, 0, w)

    r0, r1 = ys.min(), ys.max() + 1
    c0, c1 = xs.min(), xs.max() + 1

    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(h, r1 + margin)
    c1 = min(w, c1 + margin)

    return x[r0:r1, c0:c1], (r0, r1, c0, c1)


# -----------------------------
# Conversion: NIfTI -> shard (.npz)
# -----------------------------
def convert_one_subject(
    nifti_path: Path,
    out_dir: Path,
    axis: int,
    keep: Tuple[float, float],
    min_content: float,
    crop_head: bool,
    head_margin: int,
    head_dilate: int,
    head_close: int,
    max_black_ratio: float,
    black_thresh: float,
    debug_jpg: bool,
    debug_jpg_min: int,
    debug_jpg_max: int,
    debug_jpg_dir: Path,
    debug_jpg_quality: int,
    debug_seed: int,
) -> Dict[str, Any]:
    """
    Output:
      out_dir/<sid>.npz with:
        - images: (N,288,288) float32 in [0,1]
        - slice_indices: (N,) int32 (original slice index)
    Also can export debug JPG samples (after final resize).
    """
    t0 = time.time()
    sid = get_subject_id(nifti_path)
    out_path = out_dir / f"{sid}.npz"

    img = nib.load(str(nifti_path))
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    vol = robust_norm_01(vol, 1, 99)

    n = vol.shape[axis]
    s0, s1 = int(n * keep[0]), int(n * keep[1])

    slices: List[np.ndarray] = []
    slice_indices: List[int] = []

    skipped_black = 0

    for s in range(s0, s1):
        sl = np.take(vol, s, axis=axis)  # 2D [0,1]

        # filter extremely empty slices quickly
        if float((sl > 0.0).mean()) < min_content:
            continue

        if crop_head:
            sl, _bbox = head_mask_and_crop(
                sl,
                margin=head_margin,
                dilate_iter=head_dilate,
                close_iter=head_close,
            )

        # tighten crop ตัดขอบดำให้ชิด (ลดมุมดำ/ขอบดำ)
        sl = tight_intensity_crop(sl, black_thresh=black_thresh, margin=1)
        sl = resize2d_01(sl, size=OUT_SIZE)


        # ✅ NEW: skip slices with too much black background
        br = black_ratio(sl, black_thresh=black_thresh)
        if br > max_black_ratio:
            skipped_black += 1
            continue

        slices.append(sl.astype(np.float32))
        slice_indices.append(int(s))

    if len(slices) == 0:
        return {
            "subject_id": sid,
            "status": "skipped_empty_or_black",
            "n_slices": 0,
            "skipped_black": int(skipped_black),
            "in_file": str(nifti_path),
            "out_file": None,
            "elapsed_sec": time.time() - t0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack(slices, axis=0)  # (N,288,288)
    idx_arr = np.array(slice_indices, dtype=np.int32)
    np.savez_compressed(out_path, images=arr, slice_indices=idx_arr)

    # Debug JPG sampling (after final resize)
    debug_saved = 0
    if debug_jpg:
        rng = random.Random(debug_seed ^ (hash(sid) & 0xFFFFFFFF))
        k = rng.randint(int(debug_jpg_min), int(debug_jpg_max))
        k = min(k, arr.shape[0])
        picks = rng.sample(range(arr.shape[0]), k=k) if k > 0 else []

        sid_dir = debug_jpg_dir / sid
        sid_dir.mkdir(parents=True, exist_ok=True)

        for j, pidx in enumerate(picks):
            sidx = int(idx_arr[pidx])
            br = black_ratio(arr[pidx], black_thresh=black_thresh)
            out_jpg = sid_dir / f"{sid}_sample{j:02d}_slice{sidx:04d}_br{br:.2f}.jpg"
            save_jpg_from_01(arr[pidx], out_jpg, quality=debug_jpg_quality)
            debug_saved += 1

    return {
        "subject_id": sid,
        "status": "ok",
        "n_slices": int(arr.shape[0]),
        "shape": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        "in_file": str(nifti_path),
        "out_file": str(out_path),
        "elapsed_sec": time.time() - t0,
        "mtime_in": os.path.getmtime(nifti_path),
        "size_in": os.path.getsize(nifti_path),
        "mtime_out": os.path.getmtime(out_path),
        "size_out": os.path.getsize(out_path),
        "crop_head": bool(crop_head),
        "scipy_ok": bool(SCIPY_OK),
        "debug_jpg_saved": int(debug_saved) if debug_jpg else 0,
        "skipped_black": int(skipped_black),
        "max_black_ratio": float(max_black_ratio),
        "black_thresh": float(black_thresh),
    }


# -----------------------------
# Main
# -----------------------------
def main(
    zip_path: Path,
    extract_dir: Path,
    out_dir: Path,
    axis: int,
    keep: Tuple[float, float],
    min_content: float,
    crop_head: bool,
    head_margin: int,
    head_dilate: int,
    head_close: int,
    max_black_ratio: float,
    black_thresh: float,
    debug_jpg: bool,
    debug_jpg_min: int,
    debug_jpg_max: int,
    debug_jpg_dir: Path,
    debug_jpg_quality: int,
    debug_seed: int,
) -> None:
    # 1) unzip if needed
    if already_extracted(extract_dir):
        log(f"Detected extracted data in: {extract_dir} (NIfTI found). Skipping unzip.")
    else:
        log(f"No extracted NIfTI found in: {extract_dir}")
        safe_unzip(zip_path, extract_dir)

    # 2) gather nifti
    nifti_files = find_any_nifti(extract_dir)
    if len(nifti_files) == 0:
        raise RuntimeError(f"No NIfTI found under {extract_dir} after extraction.")

    log(f"Found NIfTI files: {len(nifti_files)}")
    if crop_head:
        log(f"Head crop: ENABLED | margin={head_margin} dilate={head_dilate} close={head_close} | scipy={SCIPY_OK}")
        if not SCIPY_OK:
            log("WARNING: scipy not available -> using fallback crop (less stable). Install scipy for best results.")

    log(f"Black-slice filter: max_black_ratio={max_black_ratio} black_thresh={black_thresh}")

    # 3) manifest cache
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_cc359_shards_288.json"
    manifest = load_manifest(manifest_path)

    # 4) convert
    converted = 0
    skipped_cached = 0
    empty = 0
    total_skipped_black = 0

    for i, p in enumerate(nifti_files, 1):
        sid = get_subject_id(p)
        out_path = out_dir / f"{sid}.npz"

        if should_skip(p, out_path, manifest):
            skipped_cached += 1
            if i % 25 == 0 or i == len(nifti_files):
                log(f"Progress {i}/{len(nifti_files)} | cached={skipped_cached} converted={converted} empty={empty} black_skipped={total_skipped_black}")
            continue

        info = convert_one_subject(
            nifti_path=p,
            out_dir=out_dir,
            axis=axis,
            keep=keep,
            min_content=min_content,
            crop_head=crop_head,
            head_margin=head_margin,
            head_dilate=head_dilate,
            head_close=head_close,
            max_black_ratio=max_black_ratio,
            black_thresh=black_thresh,
            debug_jpg=debug_jpg,
            debug_jpg_min=debug_jpg_min,
            debug_jpg_max=debug_jpg_max,
            debug_jpg_dir=debug_jpg_dir,
            debug_jpg_quality=debug_jpg_quality,
            debug_seed=debug_seed,
        )

        total_skipped_black += int(info.get("skipped_black", 0))

        manifest.setdefault("subjects", {})[sid] = info
        save_manifest(manifest_path, manifest)

        if info["status"] == "ok":
            converted += 1
        else:
            empty += 1

        log(f"{i}/{len(nifti_files)} {sid}: {info['status']} slices={info.get('n_slices')} skipped_black={info.get('skipped_black',0)} debug_jpg={info.get('debug_jpg_saved',0)}")

    log("==== SUMMARY ====")
    log(f"Output shards dir: {out_dir}")
    if debug_jpg:
        log(f"Debug JPG dir: {debug_jpg_dir} (per subject {debug_jpg_min}-{debug_jpg_max} images)")
    log(f"Converted: {converted} | Skipped (cached): {skipped_cached} | Empty: {empty}")
    log(f"Total skipped (black filter): {total_skipped_black}")
    log(f"Manifest: {manifest_path}")
    log("Done.")

def save_png16_from_01(img01: np.ndarray, out_path: Path) -> None:
    """
    Save float [0,1] image as 16-bit PNG (quality ดีกว่า JPG)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr16 = (np.clip(img01, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(arr16, mode="I;16").save(str(out_path), format="PNG")


def export_npz_shard_to_images(
    npz_path: Path,
    out_dir: Path,
    fmt: str = "jpg",
    quality: int = 95,
    prefix: str | None = None,
) -> int:
    """
    Export ALL slices in one shard npz (e.g., CC0001.npz) to images.
    Expects key: 'images' shape (N,H,W) float32 [0,1]
    Optional key: 'slice_indices' shape (N,)
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    if "images" not in data:
        raise KeyError(f"'images' key not found in {npz_path}. Found keys: {list(data.keys())}")

    images = data["images"]  # (N,H,W)
    slice_indices = data["slice_indices"] if "slice_indices" in data else None

    if images.ndim != 3:
        raise ValueError(f"Expected images shape (N,H,W), got {images.shape}")

    sid = prefix if prefix is not None else npz_path.stem
    n = images.shape[0]

    fmt = fmt.lower()
    for i in range(n):
        img01 = images[i].astype(np.float32)

        if slice_indices is not None:
            sidx = int(slice_indices[i])
            name = f"{sid}_slice{sidx:04d}"
        else:
            name = f"{sid}_idx{i:04d}"

        if fmt in ("jpg", "jpeg"):
            save_jpg_from_01(img01, out_dir / f"{name}.jpg", quality=quality)
        elif fmt == "png":
            save_png16_from_01(img01, out_dir / f"{name}.png")
        else:
            raise ValueError("fmt must be 'jpg' or 'png'")

    return int(n)



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", type=str, default="Original.zip")
    ap.add_argument("--extract_dir", type=str, default="data/cc359_original")
    ap.add_argument("--out_dir", type=str, default="processed/cc359_shards_288")

    ap.add_argument("--axis", type=int, default=DEFAULT_AXIS)
    ap.add_argument("--keep0", type=float, default=DEFAULT_KEEP[0])
    ap.add_argument("--keep1", type=float, default=DEFAULT_KEEP[1])
    ap.add_argument("--min_content", type=float, default=DEFAULT_MIN_CONTENT)

    # Head crop options (skull+brain)
    ap.add_argument("--crop_head", action="store_true", help="Crop skull+brain region to remove black background.")
    ap.add_argument("--head_margin", type=int, default=12)
    ap.add_argument("--head_dilate", type=int, default=2)
    ap.add_argument("--head_close", type=int, default=2)

    # ✅ Black-slice filter
    ap.add_argument("--max_black_ratio", type=float, default=0.60,
                    help="Skip slice if black pixels ratio > this value. (e.g., 0.60)")
    ap.add_argument("--black_thresh", type=float, default=0.02,
                    help="Pixel <= this (in [0,1]) is considered black. (e.g., 0.02)")

    # Debug JPG sampling
    ap.add_argument("--debug_jpg", action="store_true", help="Export random JPG samples for recheck.")
    ap.add_argument("--debug_jpg_min", type=int, default=20)
    ap.add_argument("--debug_jpg_max", type=int, default=30)
    ap.add_argument("--debug_jpg_dir", type=str, default="debug_samples/cc359_shards_288")
    ap.add_argument("--debug_jpg_quality", type=int, default=95)
    ap.add_argument("--debug_seed", type=int, default=1234)

    # Export one shard npz to ALL images
    ap.add_argument("--export_npz", type=str, default="",
                    help="If set, export this shard .npz to images then exit. Example: processed/cc359_shards_288/CC0001.npz")
    ap.add_argument("--export_out_dir", type=str, default="exports",
                    help="Output directory for exported images.")
    ap.add_argument("--export_fmt", type=str, default="jpg", choices=["jpg", "png"],
                    help="Export image format. jpg=small, png=16bit better quality.")
    ap.add_argument("--export_quality", type=int, default=95,
                    help="JPG quality (only used when --export_fmt=jpg).")


    args = ap.parse_args()

    # If user requests export, do it and exit (no conversion).
    if args.export_npz:
        npz_path = Path(args.export_npz)
        out_dir = Path(args.export_out_dir) / npz_path.stem
        n = export_npz_shard_to_images(
            npz_path=npz_path,
            out_dir=out_dir,
            fmt=args.export_fmt,
            quality=args.export_quality,
        )
        print(f"Exported {n} images from {npz_path} -> {out_dir}")
        raise SystemExit(0)


    main(
        zip_path=Path(args.zip_path),
        extract_dir=Path(args.extract_dir),
        out_dir=Path(args.out_dir),
        axis=args.axis,
        keep=(args.keep0, args.keep1),
        min_content=args.min_content,
        crop_head=bool(args.crop_head),
        head_margin=int(args.head_margin),
        head_dilate=int(args.head_dilate),
        head_close=int(args.head_close),
        max_black_ratio=float(args.max_black_ratio),
        black_thresh=float(args.black_thresh),
        debug_jpg=bool(args.debug_jpg),
        debug_jpg_min=int(args.debug_jpg_min),
        debug_jpg_max=int(args.debug_jpg_max),
        debug_jpg_dir=Path(args.debug_jpg_dir),
        debug_jpg_quality=int(args.debug_jpg_quality),
        debug_seed=int(args.debug_seed),
    )
