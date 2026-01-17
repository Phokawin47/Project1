from __future__ import annotations
from pathlib import Path
from typing import Any, Tuple

import torch
from PIL import Image


def _to_01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.numel() == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    # if likely [-1,1] -> map to [0,1]
    if mn < -0.1 and mx <= 1.1:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _chw_to_pil(img: torch.Tensor) -> Image.Image:
    img = _to_01(img).cpu()
    c, h, w = img.shape

    if c == 1:
        arr = (img[0] * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="L")
    else:
        arr = (img[:3].permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="RGB")


def _concat_triplet_h(a: Image.Image, b: Image.Image, c: Image.Image) -> Image.Image:
    mode = a.mode
    w, h = a.size
    b = b.convert(mode).resize((w, h))
    c = c.convert(mode).resize((w, h))
    out = Image.new(mode, (w * 3, h))
    out.paste(a, (0, 0))
    out.paste(b, (w, 0))
    out.paste(c, (2 * w, 0))
    return out


def _extract_pair(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    # (noisy, clean)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    # dict: {"noisy":..., "clean":...} หรือ {"input":..., "target":...}
    if isinstance(batch, dict):
        for xk, yk in [
            ("noisy", "clean"),
            ("input", "target"),
            ("x", "y"),
            ("img_noisy", "img_clean"),
            ("image_noisy", "image_clean"),
        ]:
            if xk in batch and yk in batch:
                return batch[xk], batch[yk]

        vals = [v for v in batch.values() if torch.is_tensor(v)]
        if len(vals) >= 2:
            return vals[0], vals[1]

    raise ValueError("Cannot extract (noisy, clean) from batch; please adjust _extract_pair().")


@torch.no_grad()
def save_val_triplets(
    *,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoise: torch.Tensor,
    run_dir: Path,
    epoch: int,
    max_items: int = 8,
    prefix: str = "val",
) -> None:
    out_dir = Path(run_dir) / "val_images" / f"epoch_{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    b = int(noisy.shape[0])
    n = min(max_items, b)

    for i in range(n):
        pil_clean = _chw_to_pil(clean[i])
        pil_noisy = _chw_to_pil(noisy[i])
        pil_den   = _chw_to_pil(denoise[i])
        trip = _concat_triplet_h(pil_clean, pil_noisy, pil_den)
        trip.save(out_dir / f"{prefix}_{i:03d}.png")


@torch.no_grad()
def save_val_from_loader(
    *,
    forward_fn,          # callable(noisy)->denoise
    val_loader,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    max_items: int = 8,
    prefix: str = "val",
) -> None:
    batch = next(iter(val_loader))
    noisy, clean = _extract_pair(batch)

    noisy = noisy.to(device, non_blocking=True)
    clean = clean.to(device, non_blocking=True)

    denoise = forward_fn(noisy)
    if isinstance(denoise, (tuple, list)):
        denoise = denoise[0]

    save_val_triplets(
        clean=clean.detach(),
        noisy=noisy.detach(),
        denoise=denoise.detach(),
        run_dir=run_dir,
        epoch=epoch,
        max_items=max_items,
        prefix=prefix,
    )
