"""Render a peloton race as an animated GIF or MP4.

Reuses the same `draw_road` renderer the Solara app uses, so the output
looks identical to what you'd see live — just stepped and saved instead of
shown interactively.

Output format is picked from the extension: `.gif` uses Pillow, `.mp4` uses
ffmpeg (h264). MP4 is the better choice for sending over WhatsApp etc. —
GIFs sent as "photo" attachments often get flattened/recompressed, while
MP4s are sent as proper animated video.

Usage:
    uv run python make_gif.py race.gif
    uv run python make_gif.py race.mp4
    uv run python make_gif.py trained.mp4 --population population.json
    uv run python make_gif.py race.gif --max-steps 300 --stride 2 --fps 20 --seed 7

Notes:
    - `--population population.json` loads the coefficients written by
      `main.py learn`, so you can compare a naive race to a post-learning one.
    - `--stride N` only renders every Nth step (smaller/faster file, same
      underlying simulation resolution).
    - GIF export requires Pillow (`pip install pillow` / `uv add pillow`).
    - MP4 export requires ffmpeg available on PATH (e.g. `apt install ffmpeg`
      / `brew install ffmpeg`).
"""

import argparse
import json

from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from peloton.viz import draw_road
from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def build_model(population_path: str | None, seed: int | None) -> PelotonModel:
    population = None
    if population_path:
        with open(population_path) as fh:
            population = json.load(fh)
    config = PelotonConfig(seed=seed)
    return PelotonModel(config=config, population=population)


def make_gif(
    out_path: str,
    population_path: str | None = None,
    max_steps: int = 200,
    seed: int | None = None,
    fps: int = 15,
    stride: int = 1,
) -> None:
    model = build_model(population_path, seed)

    fig = Figure(figsize=(10, 2.5))
    ax = fig.add_subplot()

    # Lazily steps the model one frame at a time, in sync with rendering, so
    # the GIF stops as soon as the race finishes (or max_steps is hit) rather
    # than pre-computing every frame up front.
    def frame_indices():
        step = 0
        yield step
        while model.running and step < max_steps:
            for _ in range(stride):
                if not model.running:
                    break
                model.step()
                step += 1
            yield step

    def update(_frame):
        ax.clear()
        draw_road(model, ax)
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        cache_frame_data=False,
    )

    if out_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
    else:
        writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    print(
        f"Wrote {out_path} ({model.steps} steps simulated, {model.n_finished} finished)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a peloton race to a GIF or MP4"
    )
    parser.add_argument("out", help="output path, e.g. race.gif or race.mp4")
    parser.add_argument(
        "--population",
        default=None,
        help="population.json from `main.py learn`, to render a post-learning race",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--stride", type=int, default=1, help="render every Nth simulated step"
    )
    args = parser.parse_args()

    make_gif(
        args.out,
        population_path=args.population,
        max_steps=args.max_steps,
        seed=args.seed,
        fps=args.fps,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
