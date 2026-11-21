"""Idol single-image -> 3D Gaussian Splat (.ply) inference runner (CUDA / MPS / CPU)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config


def parse_args():
    p = argparse.ArgumentParser(description="Idol: one image -> 3DGS object model")
    p.add_argument("--image", required=True, help="input image (render/generated -> synthetic, photo -> real)")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--out", default=str(config.OUTPUTS_DIR), help="output folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_strength", type=float, default=3.0)
    p.add_argument("--rescale_t", type=float, default=3.0)
    p.add_argument("--num_gaussians", type=int, default=262144, help="32768..1048576, multiple of 32")
    p.add_argument("--weights", default=None, help="local Idol weights dir or HF repo id (default: auto)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--dinov3_dtype", default=None, choices=["bf16", "fp16", "fp32"],
                   help="DINOv3 dtype (default: auto by device)")
    p.add_argument("--no_compile", action="store_true", help="skip torch.compile")
    p.add_argument("--compile", action="store_true", help="enable torch.compile on CUDA")
    p.add_argument("--token", default=None, help="Hugging Face token (optional, overrides HF_TOKEN)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_gaussians % 32 != 0 or not (32768 <= args.num_gaussians <= 1048576):
        sys.exit("--num_gaussians must be a multiple of 32 in [32768, 1048576]")

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    from idol.pipeline import IdolPipeline

    weights_src = args.weights or config.resolve_idol_weights()
    encoders = config.resolve_image_encoders()
    device, dinov3_dtype, optimize = config.resolve_device_and_compile(
        requested_device=args.device,
        requested_dtype=args.dinov3_dtype,
        no_compile=args.no_compile,
        force_compile=args.compile,
    )

    t0 = time.time()
    pipeline = IdolPipeline.from_pretrained(
        weights_src,
        device=device,
        dinov3_dtype=dinov3_dtype,
        optimize=optimize,
        image_encoders=encoders,
    )
    print(f"[load] {time.time() - t0:.1f}s  device={pipeline.device}  dtype={dinov3_dtype}  optimize={optimize}")

    t1 = time.time()
    result = pipeline(
        image=args.image,
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        rescale_t=args.rescale_t,
        num_gaussians=args.num_gaussians,
        verbose=True,
    )
    sample_s = time.time() - t1

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    ply_path = os.path.join(out_dir, f"{stem}.ply")
    result.gaussian.save_ply(ply_path)

    cond_path = os.path.join(out_dir, f"{stem}_condition.png")
    result.preprocessed_image.save(cond_path)

    meta_path = os.path.join(out_dir, f"{stem}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "image": os.path.abspath(args.image),
            "mode": args.mode,
            "seed": args.seed,
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
            "rescale_t": args.rescale_t,
            "num_gaussians": args.num_gaussians,
            "camera": result.camera,
            "load_seconds": round(time.time() - t0, 2),
            "sample_seconds": round(sample_s, 2),
            "device": str(pipeline.device),
        }, f, indent=2, ensure_ascii=False)

    print(f"[done] {sample_s:.1f}s sampling")
    print(f"  ply       {ply_path}  ({os.path.getsize(ply_path) / 1e6:.1f} MB)")
    print(f"  condition {cond_path}")
    print(f"  meta      {meta_path}")
    print(f"  camera    {result.camera}")


if __name__ == "__main__":
    main()
