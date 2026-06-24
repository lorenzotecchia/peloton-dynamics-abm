"""Merge plot_leading_pack and plot_rank_detail outputs into one tall image.

Generates the source images if they don't already exist, then stacks them
vertically in this order:
  1. leading_pack_size.png
  2. rank{N}_stamina_vs_pull.png  (for each requested rank, top to bottom)

Usage:
    uv run python scripts/plot_merge_overview.py --dir analysis_output/v-and-v
    uv run python scripts/plot_merge_overview.py --dir analysis_output/v-and-v --ranks 1 2 3
"""
import argparse
import os
import subprocess
import sys

from PIL import Image


def ensure(path: str, cmd: list[str]) -> str:
    if not os.path.exists(path):
        print(f"Generating {path} ...")
        subprocess.run(cmd, check=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",   default="analysis_output/v-and-v")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    p.add_argument("--out",   default=None)
    args = p.parse_args()

    d   = args.dir
    out = args.out or os.path.join(d, "overview.png")

    python = [sys.executable]

    sources = [
        ensure(
            os.path.join(d, "leading_pack_size.png"),
            python + ["scripts/plot_leading_pack.py", "--dir", d],
        )
    ]
    for rank in args.ranks:
        sources.append(
            ensure(
                os.path.join(d, f"rank{rank}_stamina_vs_pull.png"),
                python + ["scripts/plot_rank_detail.py", "--dir", d, "--rank", str(rank)],
            )
        )

    imgs   = [Image.open(s) for s in sources]
    width  = max(im.width for im in imgs)
    height = sum(im.height for im in imgs)

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    y = 0
    for im in imgs:
        # centre narrower images horizontally
        x = (width - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height

    canvas.save(out, dpi=(150, 150))
    print(f"Saved {out}  ({width}×{height} px)")


if __name__ == "__main__":
    main()
