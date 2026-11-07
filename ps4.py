"""PS4: edge-preserving depth-map upsampling on NYU Depth V2."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parent
DMAX = 10.0
HEIGHT, WIDTH = 240, 320
FACTORS = (2, 4, 8)
METHODS = (
    ("interpolation", "nearest"),
    ("interpolation", "bilinear"),
    ("interpolation", "bicubic"),
    ("classical_edge_aware", "joint_bilateral"),
    ("classical_edge_aware", "guided_filter"),
    ("CNN-only", "cnn"),
    ("Hybrid", "hybrid"),
)


def _find_hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    matches: list[h5py.Dataset] = []

    def collect(path: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset) and path.rsplit("/", 1)[-1].lower() == name:
            matches.append(item)

    handle.visititems(collect)
    if not matches:
        raise KeyError(f"Could not find an HDF5 dataset named {name!r} in {handle.filename}")
    return min(matches, key=lambda dataset: dataset.name.count("/"))


def _sample_hdf5_item(dataset: h5py.Dataset, sample_axis: int, index: int) -> np.ndarray:
    selector: list[Any] = [slice(None)] * dataset.ndim
    selector[sample_axis] = index
    return np.asarray(dataset[tuple(selector)])


def _orient_image(image: np.ndarray, channel_axis: int) -> np.ndarray:
    image = np.moveaxis(image, channel_axis, -1)
    if image.shape[0] > image.shape[1]:
        image = np.swapaxes(image, 0, 1)
    return image


def _orient_depth(depth: np.ndarray) -> np.ndarray:
    if depth.shape[0] > depth.shape[1]:
        depth = depth.T
    return depth


def prepare_nyu(mat_path: Path, output_path: Path, samples: int = 500) -> Path:
    """Read evenly spaced NYU frames from a v7.3 MAT file and resize them."""
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"NYU MAT file not found: {mat_path}. Place nyu_depth_v2_labeled.mat in data/raw/ "
            "or pass --mat PATH."
        )
    if samples < 1:
        raise ValueError("--samples must be positive")

    with h5py.File(mat_path, "r") as handle:
        image_ds = _find_hdf5_dataset(handle, "images")
        depth_ds = _find_hdf5_dataset(handle, "depths")
        if image_ds.ndim != 4 or depth_ds.ndim != 3:
            raise ValueError(
                f"Expected images/depths with 4/3 dimensions, got "
                f"{image_ds.shape} and {depth_ds.shape}"
            )
        channel_axes = [axis for axis, size in enumerate(image_ds.shape) if size == 3]
        if not channel_axes:
            raise ValueError(f"No 3-channel axis found in image data shape {image_ds.shape}")
        channel_axis = channel_axes[0]
        image_sample_axis = max(
            (axis for axis in range(image_ds.ndim) if axis != channel_axis),
            key=lambda axis: image_ds.shape[axis],
        )
        depth_sample_axis = max(range(depth_ds.ndim), key=lambda axis: depth_ds.shape[axis])
        image_count = image_ds.shape[image_sample_axis]
        depth_count = depth_ds.shape[depth_sample_axis]
        if image_count != depth_count:
            raise ValueError(
                f"Image/depth sample counts do not match ({image_count} vs {depth_count})"
            )
        indices = np.linspace(0, image_count - 1, min(samples, image_count), dtype=np.int64)
        images = np.empty((len(indices), HEIGHT, WIDTH, 3), dtype=np.uint8)
        depths = np.empty((len(indices), HEIGHT, WIDTH), dtype=np.float32)

        for out_index, source_index in enumerate(indices):
            image = _sample_hdf5_item(image_ds, image_sample_axis, int(source_index))
            adjusted_channel_axis = channel_axis - (image_sample_axis < channel_axis)
            image = _orient_image(image, int(adjusted_channel_axis))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected RGB image, got sampled shape {image.shape}")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            images[out_index] = cv2.resize(
                image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
            )

            depth = _orient_depth(
                _sample_hdf5_item(depth_ds, depth_sample_axis, int(source_index))
            )
            depth = cv2.resize(
                depth.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            valid = np.isfinite(depth) & (depth > 0)
            depths[out_index] = np.where(valid, np.minimum(depth, DMAX), 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, images=images, depths=depths, indices=indices)
    print(f"Saved {len(indices)} RGB/depth pairs to {output_path}")
    return output_path


def load_subset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed subset not found: {path}. Run `python ps4.py prepare` first."
        )
    with np.load(path) as data:
        if "images" not in data or "depths" not in data:
            raise ValueError(f"{path} must contain 'images' and 'depths' arrays")
        images = data["images"]
        depths = data["depths"]
    if images.ndim != 4 or images.shape[-1] != 3 or depths.ndim != 3:
        raise ValueError(f"Invalid subset shapes: images={images.shape}, depths={depths.shape}")
    if images.shape[:3] != depths.shape:
        raise ValueError(f"RGB/depth dimensions differ: {images.shape} vs {depths.shape}")
    if images.shape[1:3] != (HEIGHT, WIDTH):
        raise ValueError(
            f"Expected {HEIGHT}x{WIDTH} frames, found {images.shape[1:3]}"
        )
    return images, depths.astype(np.float32, copy=False)


def split_indices(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(count * 0.70)
    val_end = train_end + int(count * 0.10)
    return (
        np.arange(0, train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, count),
    )


def _downsample(depth: np.ndarray, factor: int) -> np.ndarray:
    height, width = depth.shape
    return cv2.resize(
        depth,
        (max(1, width // factor), max(1, height // factor)),
        interpolation=cv2.INTER_AREA,
    )


def _add_noise(depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = depth + rng.normal(0.0, 0.02, depth.shape).astype(np.float32)
    spikes = rng.random(depth.shape) < 0.05
    noisy[spikes] = rng.uniform(0.0, DMAX, int(spikes.sum())).astype(np.float32)
    return np.clip(noisy, 0.0, DMAX)


def upsample(depth: np.ndarray, shape: tuple[int, int], method: str) -> np.ndarray:
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    if method not in interpolation:
        raise ValueError(f"Unknown interpolation method: {method}")
    height, width = shape
    return cv2.resize(depth, (width, height), interpolation=interpolation[method])


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int) -> np.ndarray:
    """Single-channel guided filter with a normalized box filter."""
    guide = np.asarray(guide, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    window = (2 * radius + 1, 2 * radius + 1)

    def box(value: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            value, ddepth=-1, ksize=window, normalize=True, borderType=cv2.BORDER_REFLECT
        )

    mean_guide = box(guide)
    mean_source = box(source)
    corr_guide = box(guide * guide)
    corr_cross = box(guide * source)
    variance = corr_guide - mean_guide * mean_guide
    covariance = corr_cross - mean_guide * mean_source
    coefficient = covariance / (variance + 1e-3)
    intercept = mean_source - coefficient * mean_guide
    return box(coefficient) * guide + box(intercept)


def edge_mask(gt: np.ndarray, thr: float = 0.5, dilate: int = 7) -> np.ndarray:
    """Return a dilated mask around depth discontinuities."""
    if dilate < 1:
        raise ValueError("dilate must be positive")
    depth = np.asarray(gt, dtype=np.float32)
    gradient_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.hypot(gradient_x, gradient_y) > thr
    kernel = np.ones((dilate, dilate), dtype=np.uint8)
    return cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)


def _make_inputs(
    rgb: np.ndarray,
    ground_truth: np.ndarray,
    factor: int,
    noise: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = _downsample(ground_truth, factor)
    if noise:
        low = _add_noise(low, rng)
    shape = ground_truth.shape
    bicubic = upsample(low, shape, "bicubic")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    guided = guided_filter(gray, bicubic / DMAX, max(4, factor)) * DMAX
    return bicubic, np.clip(guided, 0.0, DMAX), gray


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.enc1(value)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        middle = self.bottleneck(self.pool(third))
        value = self.dec3(torch.cat((self._up(middle, third), third), dim=1))
        value = self.dec2(torch.cat((self._up(value, second), second), dim=1))
        value = self.dec1(torch.cat((self._up(value, first), first), dim=1))
        return self.output(value)


def _model_input(
    rgb: np.ndarray, bicubic: np.ndarray, guided: np.ndarray, family: str
) -> tuple[np.ndarray, np.ndarray]:
    color = rgb.astype(np.float32) / 255.0
    bicubic_norm = bicubic.astype(np.float32) / DMAX
    if family == "hybrid":
        channels = np.concatenate(
            (bicubic_norm[..., None], (guided / DMAX)[..., None], color), axis=-1
        )
        return channels, guided
    channels = np.concatenate((bicubic_norm[..., None], color), axis=-1)
    return channels, bicubic


def _to_tensor(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(device)


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target > 0
    if not mask.any():
        return F.l1_loss(prediction, target)
    pixel_loss = torch.abs(prediction - target)[mask].mean()
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    grad_terms: list[torch.Tensor] = []
    if mask_x.any():
        grad_terms.append(torch.abs(pred_x - target_x)[mask_x].mean())
    if mask_y.any():
        grad_terms.append(torch.abs(pred_y - target_y)[mask_y].mean())
    gradient_loss = torch.stack(grad_terms).mean() if grad_terms else pixel_loss.new_zeros(())
    return pixel_loss + 2.0 * gradient_loss


def _save_checkpoint(path: Path, model: UNet, family: str, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "family": family,
            "dmax": DMAX,
            "input_channels": 5 if family == "hybrid" else 4,
            "step": step,
        },
        path,
    )


def train_model(
    family: str,
    images: np.ndarray,
    depths: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    minutes: float,
    batch_size: int,
    noise: bool,
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> Path:
    if family not in ("hybrid", "cnn"):
        raise ValueError(f"Unsupported training family: {family}")
    if minutes <= 0:
        raise ValueError("--minutes must be positive")
    if len(train_indices) == 0:
        raise ValueError("The training split is empty; provide more NYU frames.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = UNet(5 if family == "hybrid" else 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    duration = minutes * 60.0
    start = time.monotonic()
    deadline = start + duration
    step = 0
    last_checkpoint = 0.0
    checkpoint_path = checkpoint_dir / f"ckpt_{family}.pt"
    model.train()
    print(f"Training {family} for up to {minutes:g} minute(s) on {device}.")

    while True:
        batch_indices = rng.choice(train_indices, size=batch_size, replace=True)
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index in batch_indices:
            rgb = images[int(index)]
            target = depths[int(index)]
            factor = int(rng.choice(FACTORS))
            bicubic, guided, _ = _make_inputs(rgb, target, factor, noise, rng)
            features, base = _model_input(rgb, bicubic, guided, family)
            if rng.random() < 0.5:
                features = features[:, ::-1, :].copy()
                base = base[:, ::-1].copy()
                target = target[:, ::-1].copy()
            inputs.append(features)
            targets.append(np.stack((base / DMAX, target / DMAX), axis=-1))

        feature_tensor = _to_tensor(np.stack(inputs), device)
        target_tensor = _to_tensor(np.stack(targets), device)
        base_tensor = target_tensor[:, :1]
        ground_truth_tensor = target_tensor[:, 1:2]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = (base_tensor + model(feature_tensor)).clamp(0.0, 1.0)
        loss = _masked_loss(prediction, ground_truth_tensor)
        loss.backward()
        optimizer.step()

        elapsed = time.monotonic() - start
        progress = min(elapsed / duration, 1.0)
        learning_rate = 2e-3 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        step += 1

        if step == 1 or elapsed - last_checkpoint >= 60:
            _save_checkpoint(checkpoint_path, model, family, step)
            last_checkpoint = elapsed
        if step % 10 == 0:
            print(f"  step={step} loss={loss.item():.5f} elapsed={elapsed:.1f}s")
        if time.monotonic() >= deadline:
            break

    _save_checkpoint(checkpoint_path, model, family, step)
    if len(val_indices):
        score = _validation_loss(
            model, family, images, depths, val_indices[: min(16, len(val_indices))], device
        )
        print(f"  validation loss={score:.5f}")
    print(f"Saved {checkpoint_path} ({step} steps)")
    return checkpoint_path


def _validation_loss(
    model: UNet,
    family: str,
    images: np.ndarray,
    depths: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(2026)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for index in indices:
        rgb, target = images[int(index)], depths[int(index)]
        factor = FACTORS[len(feature_rows) % len(FACTORS)]
        bicubic, guided, _ = _make_inputs(rgb, target, factor, False, rng)
        features, base = _model_input(rgb, bicubic, guided, family)
        feature_rows.append(features)
        target_rows.append(np.stack((base / DMAX, target / DMAX), axis=-1))
    model.eval()
    with torch.no_grad():
        inputs = _to_tensor(np.stack(feature_rows), device)
        values = _to_tensor(np.stack(target_rows), device)
        prediction = (values[:, :1] + model(inputs)).clamp(0.0, 1.0)
        result = float(_masked_loss(prediction, values[:, 1:2]).item())
    model.train()
    return result


def _load_model(path: Path, family: str, device: torch.device) -> UNet:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run `python ps4.py train` first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("family") != family or checkpoint.get("dmax") != DMAX:
        raise ValueError(f"{path} does not contain a compatible {family} checkpoint")
    model = UNet(int(checkpoint["input_channels"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict(
    model: UNet, rgb: np.ndarray, bicubic: np.ndarray, guided: np.ndarray, family: str,
    device: torch.device,
) -> np.ndarray:
    features, base = _model_input(rgb, bicubic, guided, family)
    with torch.no_grad():
        prediction = model(_to_tensor(features[None], device))
        output = (prediction + torch.from_numpy(base[None, None].astype(np.float32) / DMAX).to(device))
        return output.clamp(0.0, 1.0)[0, 0].cpu().numpy() * DMAX


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
def _peak_rss_mib() -> float:
    import psutil
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / (1024.0 ** 2)  # peak_wset is Windows-only

def _predict_method(
    method: str,
    low: np.ndarray,
    rgb: np.ndarray,
    models: dict[str, UNet],
    device: torch.device,
    ximgproc_available: bool,
) -> np.ndarray:
    shape = rgb.shape[:2]
    if method == "nearest":
        return upsample(low, shape, "nearest")
    if method == "bilinear":
        return upsample(low, shape, "bilinear")

    bicubic = upsample(low, shape, "bicubic")
    if method == "bicubic":
        return bicubic

    if method == "joint_bilateral":
        if not ximgproc_available:
            raise RuntimeError("cv2.ximgproc jointBilateralFilter is unavailable")
        joint = cv2.ximgproc.jointBilateralFilter(
            rgb.astype(np.float32) / 255.0,
            (bicubic / DMAX).astype(np.float32),
            d=2 * max(4, int(round(shape[0] / low.shape[0]))) + 1,
            sigmaColor=0.1,
            sigmaSpace=float(max(4, int(round(shape[0] / low.shape[0])))),
        )
        return np.clip(joint * DMAX, 0.0, DMAX)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    factor = max(1, int(round(shape[0] / low.shape[0])))
    guided = np.clip(guided_filter(gray, bicubic / DMAX, max(4, factor)) * DMAX, 0.0, DMAX)
    if method == "guided_filter":
        return guided
    if method == "cnn":
        return _predict(models["cnn"], rgb, bicubic, guided, "cnn", device)
    if method == "hybrid":
        return _predict(models["hybrid"], rgb, bicubic, guided, "hybrid", device)
    raise ValueError(f"Unknown evaluation method: {method}")


def _metric_sums(
    prediction: np.ndarray, target: np.ndarray, region: np.ndarray | None = None
) -> tuple[float, float, float, int]:
    valid = np.isfinite(target) & (target > 0) & (target <= DMAX)
    if region is not None:
        valid &= region
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0
    error = np.asarray(prediction, dtype=np.float64)[valid] - target[valid]
    return (
        float(np.square(error).sum()),
        float(np.abs(error).sum()),
        float((np.abs(error) / target[valid]).sum()),
        int(valid.sum()),
    )


def _metric_row(
    family: str,
    method: str,
    factor: int,
    sums: tuple[float, float, float, int],
    boundary_sums: tuple[float, float, float, int],
    ssim: float,
    latency_ms: float,
    status: str,
) -> dict[str, Any]:
    squared, absolute, relative, count = sums
    boundary_squared, boundary_absolute, _, boundary_count = boundary_sums
    if count == 0:
        rmse = mae = abs_rel = psnr = float("nan")
    else:
        rmse = math.sqrt(squared / count)
        mae = absolute / count
        abs_rel = relative / count
        psnr = float("inf") if rmse == 0 else 20.0 * math.log10(DMAX / rmse)
    if boundary_count == 0:
        boundary_rmse = boundary_mae = float("nan")
    else:
        boundary_rmse = math.sqrt(boundary_squared / boundary_count)
        boundary_mae = boundary_absolute / boundary_count
    return {
        "family": family,
        "method": method,
        "factor": factor,
        "split": "test",
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "psnr": psnr,
        "boundary_rmse": boundary_rmse,
        "boundary_mae": boundary_mae,
        "ssim": ssim,
        "latency_ms": latency_ms,
        "valid_pixels": count,
        "status": status,
    }


def _write_results_markdown(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Depth Upsampling Results", ""]
    for factor in FACTORS:
        lines.extend(
            [
                f"## Factor {factor}x",
                "",
                "| Method | RMSE (m) | MAE (m) | Boundary RMSE (m) | "
                "Boundary MAE (m) | Abs Rel | PSNR (dB) | SSIM | Latency (ms/image) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row["factor"] != factor:
                continue
            values = (
                "N/A" if row["status"] != "ok" else f'{row["rmse"]:.4f}',
                "N/A" if row["status"] != "ok" else f'{row["mae"]:.4f}',
                "N/A" if not math.isfinite(row["boundary_rmse"]) else f'{row["boundary_rmse"]:.4f}',
                "N/A" if not math.isfinite(row["boundary_mae"]) else f'{row["boundary_mae"]:.4f}',
                "N/A" if row["status"] != "ok" else f'{row["abs_rel"]:.4f}',
                "N/A" if row["status"] != "ok" else f'{row["psnr"]:.2f}',
                "N/A" if not math.isfinite(row["ssim"]) else f'{row["ssim"]:.4f}',
                "N/A" if not math.isfinite(row["latency_ms"]) else f'{row["latency_ms"]:.3f}',
            )
            lines.append(f'| {row["method"]} | ' + " | ".join(values) + " |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def evaluate(
    images: np.ndarray,
    depths: np.ndarray,
    test_indices: np.ndarray,
    checkpoint_dir: Path,
    output_path: Path,
    noise: bool,
    seed: int,
    device: torch.device,
) -> Path:
    if len(test_indices) == 0:
        raise ValueError("The test split is empty; provide more NYU frames.")
    models = {
        family: _load_model(checkpoint_dir / f"ckpt_{family}.pt", family, device)
        for family in ("cnn", "hybrid")
    }
    ximgproc_available = hasattr(cv2, "ximgproc") and hasattr(
        cv2.ximgproc, "jointBilateralFilter"
    )
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Evaluation hardware: {hardware}")
    parameter_counts: dict[str, int] = {}
    peak_memory: dict[str, int] = {"cnn": 0, "hybrid": 0}
    for family, model in models.items():
        parameter_counts[family] = sum(parameter.numel() for parameter in model.parameters())
        checkpoint_path = checkpoint_dir / f"ckpt_{family}.pt"
        size_mb = checkpoint_path.stat().st_size / (1024.0 * 1024.0)
        print(
            f"{family}: parameters={parameter_counts[family]:,}, "
            f"checkpoint={size_mb:.2f} MiB"
        )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    print(f"Evaluating {len(test_indices)} test frames at factors {FACTORS}.")

    for factor in FACTORS:
        sums_by_method = {method: [0.0, 0.0, 0.0, 0] for _, method in METHODS}
        boundary_sums_by_method = {method: [0.0, 0.0, 0.0, 0] for _, method in METHODS}
        ssim_by_method = {method: 0.0 for _, method in METHODS}
        latency_by_method: dict[str, list[float]] = {method: [] for _, method in METHODS}
        lows_by_index: dict[int, np.ndarray] = {}
        for index in test_indices:
            target = depths[int(index)]
            low = _downsample(target, factor)
            lows_by_index[int(index)] = _add_noise(low, rng) if noise else low

        warmup_index = int(test_indices[0])
        for _, method in METHODS:
            if method == "joint_bilateral" and not ximgproc_available:
                continue
            if method in ("cnn", "hybrid") and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            _predict_method(
                method,
                lows_by_index[warmup_index],
                images[warmup_index],
                models,
                device,
                ximgproc_available,
            )
            _synchronize(device)
            if method in ("cnn", "hybrid") and device.type == "cuda":
                peak_memory[method] = max(
                    peak_memory[method], torch.cuda.max_memory_allocated(device)
                )

        for count, index_value in enumerate(test_indices, start=1):
            index = int(index_value)
            rgb, target = images[index], depths[index]
            low = lows_by_index[index]
            boundary = edge_mask(target)
            for family, method in METHODS:
                if method == "joint_bilateral" and not ximgproc_available:
                    continue
                _synchronize(device)
                start = time.perf_counter()
                prediction = _predict_method(
                    method, low, rgb, models, device, ximgproc_available
                )
                _synchronize(device)
                latency_by_method[method].append((time.perf_counter() - start) * 1000.0)
                if method in ("cnn", "hybrid") and device.type == "cuda":
                    peak_memory[method] = max(
                        peak_memory[method], torch.cuda.max_memory_allocated(device)
                    )

                values = _metric_sums(prediction, target)
                boundary_values = _metric_sums(prediction, target, boundary)
                total = sums_by_method[method]
                boundary_total = boundary_sums_by_method[method]
                for position in range(4):
                    total[position] += values[position]
                    boundary_total[position] += boundary_values[position]
                _, ssim_map = structural_similarity(
                    target.astype(np.float32),
                    prediction.astype(np.float32),
                    data_range=DMAX,
                    full=True,
                )
                ssim_by_method[method] += float(
                    ssim_map[(target > 0) & (target <= DMAX)].mean()
                )
            if count % 10 == 0 or count == len(test_indices):
                print(f"  factor={factor} processed {count}/{len(test_indices)}")

        for family, method in METHODS:
            available = method != "joint_bilateral" or ximgproc_available
            status = "ok" if available else "unavailable: cv2.ximgproc jointBilateralFilter missing"
            sums = tuple(sums_by_method[method]) if available else (0.0, 0.0, 0.0, 0)
            boundary_sums = (
                tuple(boundary_sums_by_method[method]) if available else (0.0, 0.0, 0.0, 0)
            )
            ssim = ssim_by_method[method] / len(test_indices) if available else float("nan")
            timings = latency_by_method[method]
            latency_ms = float(np.mean(timings)) if timings else float("nan")
            rows.append(
                _metric_row(
                    family,
                    method,
                    factor,
                    sums,
                    boundary_sums,
                    ssim,
                    latency_ms,
                    status,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "family", "method", "factor", "split", "rmse", "mae", "abs_rel", "psnr",
        "boundary_rmse", "boundary_mae", "ssim", "latency_ms", "valid_pixels", "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_path.with_name(
        "results_noisy.md" if noise else "results.md"
    )
    _write_results_markdown(rows, markdown_path)
    print(f"Saved {len(rows)} results to {output_path} and {markdown_path}")
    for row in rows:
        if row["status"] == "ok":
            print(
                f'  {row["method"]:16s} x{row["factor"]}: '
                f'RMSE={row["rmse"]:.4f} MAE={row["mae"]:.4f} '
                f'boundary-RMSE={row["boundary_rmse"]:.4f} '
                f'SSIM={row["ssim"]:.4f} latency={row["latency_ms"]:.3f} ms'
            )
    for family in ("cnn", "hybrid"):
        if device.type == "cuda":
            print(f"{family}: peak GPU memory={peak_memory[family] / (1024.0 ** 2):.2f} MiB")
        else:
            print(f"{family}: peak GPU memory=N/A (CPU)")
    print(f"Peak CPU memory (process): {_peak_rss_mib():.0f} MiB")
    return output_path


def create_figures(
    images: np.ndarray,
    depths: np.ndarray,
    test_indices: np.ndarray,
    checkpoint_dir: Path,
    output_dir: Path,
    results_path: Path,
    device: torch.device,
) -> tuple[Path, Path]:
    if len(test_indices) == 0:
        raise ValueError("The test split is empty; provide more NYU frames.")
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results CSV {results_path}; run `python ps4.py evaluate` first.")
    models = {
        family: _load_model(checkpoint_dir / f"ckpt_{family}.pt", family, device)
        for family in ("cnn", "hybrid")
    }
    ximgproc_available = hasattr(cv2, "ximgproc") and hasattr(
        cv2.ximgproc, "jointBilateralFilter"
    )
    selected_positions = np.linspace(0, len(test_indices) - 1, 3, dtype=np.int64)
    selected_indices = test_indices[selected_positions]
    figure_rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for index_value in selected_indices:
        index = int(index_value)
        rgb, target = images[index], depths[index]
        low = _downsample(target, 4)
        predictions = {
            method: _predict_method(method, low, rgb, models, device, ximgproc_available)
            for method in ("bicubic", "guided_filter", "cnn", "hybrid")
        }
        figure_rows.append((rgb, target, predictions))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_path = output_dir / "fig_qualitative_4x.png"
    figure, axes = plt.subplots(6, 7, figsize=(18, 14), constrained_layout=True)
    titles = ("RGB", "GT", "Bicubic", "Guided", "CNN-only", "Hybrid", "Boundary mask")
    for frame_number, (rgb, target, predictions) in enumerate(figure_rows):
        image_row = frame_number * 2
        image_values = (
            (rgb, None),
            (target, "viridis"),
            (predictions["bicubic"], "viridis"),
            (predictions["guided_filter"], "viridis"),
            (predictions["cnn"], "viridis"),
            (predictions["hybrid"], "viridis"),
            (edge_mask(target).astype(np.uint8), "gray"),
        )
        for column, (value, color_map) in enumerate(image_values):
            axis = axes[image_row, column]
            axis.imshow(value, cmap=color_map, vmin=0, vmax=1 if column == 6 else DMAX)
            axis.set_title(titles[column] if frame_number == 0 else "")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[image_row, 0].set_ylabel(f"Frame {int(selected_indices[frame_number])}")

        axes[image_row + 1, 0].axis("off")
        axes[image_row + 1, 1].axis("off")
        for column, method in zip(
            (2, 3, 4, 5), ("bicubic", "guided_filter", "cnn", "hybrid")
        ):
            error_axis = axes[image_row + 1, column]
            error_axis.imshow(
                np.abs(predictions[method] - target),
                cmap="magma",
                vmin=0.0,
                vmax=0.5,
            )
            error_axis.set_title("|error| (m)" if frame_number == 0 else "")
            error_axis.set_xticks([])
            error_axis.set_yticks([])
        axes[image_row + 1, 6].axis("off")
        axes[image_row + 1, 2].set_ylabel("|error|")
    figure.savefig(qualitative_path, dpi=160)
    plt.close(figure)

    with results_path.open("r", newline="", encoding="utf-8") as file:
        result_rows = list(csv.DictReader(file))
    required_columns = {"method", "factor", "rmse", "boundary_rmse", "latency_ms"}
    if not result_rows or not required_columns.issubset(result_rows[0]):
        raise ValueError(f"{results_path} is missing the metrics required for the bar chart.")
    result_map = {
        (row["method"], int(row["factor"])): row for row in result_rows
    }
    methods = [method for _, method in METHODS]
    x_values = np.arange(len(methods))
    width = 0.8 / len(FACTORS)
    metrics = (
        ("rmse", "RMSE (m)"),
        ("boundary_rmse", "Boundary RMSE (m)"),
        ("latency_ms", "Latency (ms/image)"),
    )
    bar_path = output_dir / "fig_bars.png"
    figure, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    for axis, (key, title) in zip(axes, metrics):
        for factor_position, factor in enumerate(FACTORS):
            values = []
            for method in methods:
                row = result_map.get((method, factor))
                try:
                    values.append(float(row[key]) if row is not None else float("nan"))
                except (TypeError, ValueError):
                    values.append(float("nan"))
            axis.bar(
                x_values + (factor_position - 1) * width,
                values,
                width,
                label=f"{factor}x",
            )
        axis.set_title(title)
        axis.set_xticks(x_values)
        axis.set_xticklabels(methods, rotation=35, ha="right")
        axis.legend(title="Factor")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(bar_path, dpi=160)
    plt.close(figure)
    print(f"Saved figures to {qualitative_path} and {bar_path}")
    return qualitative_path, bar_path


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    return torch.device(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, train, and compare depth-map upsampling methods."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create the evenly sampled NYU subset.")
    prepare.add_argument(
        "--mat", type=Path, default=ROOT / "data" / "raw" / "nyu_depth_v2_labeled.mat"
    )
    prepare.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "processed" / "nyu_subset.npz"
    )
    prepare.add_argument("--samples", type=int, default=500)

    for name in ("train", "evaluate", "all"):
        command = commands.add_parser(
            name, help={"train": "Train CNN and hybrid models.", "evaluate": "Evaluate all methods.",
                        "all": "Prepare data, train both models, and evaluate."}[name]
        )
        command.add_argument(
            "--dataset", type=Path, default=ROOT / "data" / "processed" / "nyu_subset.npz"
        )
        command.add_argument("--minutes", type=float, default=10.0, help="Training budget per model.")
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--noise", action="store_true", help="Add 2 cm noise and 5% outlier spikes.")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    figures = commands.add_parser("figures", help="Create qualitative and metrics figures.")
    figures.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "processed" / "nyu_subset.npz"
    )
    figures.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_nyu(args.mat, args.dataset, args.samples)
        return 0
    if args.command == "all" and not args.dataset.is_file():
        default_mat = ROOT / "data" / "raw" / "nyu_depth_v2_labeled.mat"
        prepare_nyu(default_mat, args.dataset)

    images, depths = load_subset(args.dataset)
    train_indices, val_indices, test_indices = split_indices(len(images))
    device = _device(args.device)

    if args.command == "figures":
        create_figures(
            images,
            depths,
            test_indices,
            ROOT / "checkpoints",
            ROOT / "outputs" / "figures",
            ROOT / "outputs" / "results.csv",
            device,
        )
        return 0

    if args.command in ("train", "all"):
        for family in ("cnn", "hybrid"):
            train_model(
                family,
                images,
                depths,
                train_indices,
                val_indices,
                args.minutes,
                args.batch_size,
                args.noise,
                args.seed,
                device,
                ROOT / "checkpoints",
            )
    if args.command in ("evaluate", "all"):
        evaluate(
            images,
            depths,
            test_indices,
            ROOT / "checkpoints",
            ROOT / "outputs" / ("results_noisy.csv" if args.noise else "results.csv"),
            args.noise,
            args.seed,
            device,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
