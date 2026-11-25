"""Measures one model on this PC and appends every run to a CSV.

    python bench.py --model qwen-turbo --out results
    python bench.py --model moss --sizes 512x512 1024x1024
    python bench.py --model klein --edit-sizes 1024x1024

For each size it draws the three prompts below, one after another, each with a new prompt
(so each run includes swapping in the text encoder, as when you type something new).
Nothing is averaged here: one CSV row per picture. summarise.py turns rows into medians.

What a row records: seconds for the whole picture and for each stage, the card's dedicated
memory before and at its peak (whole card, as nvidia-smi shows it), how much the card's
shared memory grew (that is spill into system RAM), system RAM before and at its peak.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

from app import monitor
from app.engine import MODELS, Engine, versions

PROMPTS = {
    "fox": "A red fox standing in fresh snow at sunrise, breath visible in the cold air, pine forest behind, photograph",
    "sign": 'A neon sign that reads "BELOW THE HEADLINE" glowing in the window of a small coffee shop at night, '
            "wet street reflections, photograph",
    "fisherman": "Portrait of an old fisherman mending a net on a wooden pier, weathered hands, overcast morning "
                 "light, 85mm photograph",
}
EDITS = {
    "night": "Make it night with a full moon in the sky. Keep the fox, its pose and the framing.",
    "scarf": "Put a red knitted scarf around the fox's neck. Keep everything else the same.",
    "watercolour": "Redraw the whole picture as a loose watercolour painting on textured paper. Keep the fox and the composition.",
}
SEED = 7

FIELDS = ["time", "model", "repo", "revision", "precision", "placement", "task", "prompt_id", "width", "height",
          "steps", "guidance", "seed", "run", "seconds", "swap_to_text_encoder_s", "text_encoder_s",
          "swap_to_transformer_s", "draw_s", "vram_start_mb", "vram_peak_mb", "vram_total_mb",
          "torch_peak_alloc_mb", "torch_peak_reserved_mb", "shared_start_mb", "shared_growth_mb",
          "ram_start_mb", "ram_start_pct", "ram_peak_mb", "ram_peak_pct", "ram_total_mb", "process_peak_mb", "commit_peak_mb",
          "error", "note", "image"]
PRECISION = "transformer and text encoder 4-bit NF4 (bitsandbytes); VAE as shipped"
PLACEMENT = "one big part on the card at a time; the idle part stays on disk, not in RAM"


def size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--sizes", nargs="*", type=size, default=[size(s) for s in
                    ("512x512", "768x768", "1024x1024", "1344x1344", "1344x768")])
    ap.add_argument("--edit-sizes", nargs="*", type=size, default=[size("1024x1024")])
    ap.add_argument("--edit-input", default="", help="picture to edit (default: this run's fox at 1024x1024)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-edit", action="store_true")
    ap.add_argument("--edit-extra", default="{}", help='JSON passed to the pipeline for edits, e.g. '
                    '{"use_kv_cache": false, "resolution": 768} (Qwen-Image: skip the key/value cache, '
                    'read the reference picture at 768 px)')
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    out = Path(a.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    spec = MODELS[a.model]
    csv_path = out / "results.csv"
    new = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        writer.writeheader()

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def row(stats: dict, **more) -> None:
        if stats.get("ram_total_mb"):
            more.setdefault("ram_start_pct", round(100 * stats["ram_start_mb"] / stats["ram_total_mb"], 1))
            more.setdefault("ram_peak_pct", round(100 * stats["ram_peak_mb"] / stats["ram_total_mb"], 1))
        writer.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "repo": spec.repo,
                         "revision": spec.revision[:12], "precision": PRECISION, "placement": PLACEMENT,
                         "guidance": spec.guidance, "note": a.note, **stats, **more})
        f.flush()

    env = {**versions(), "python": platform.python_version(), "os": platform.platform(),
           "start": monitor.snapshot(), "commit_mb": monitor.commit_mb()}
    (out / f"environment-{a.model}.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    log(f"environment: {json.dumps(env)}")

    engine = Engine(log=log)
    load = engine.load(a.model)
    for b in load["builds"]:
        if b.get("built"):
            row({k: b.get(k) for k in FIELDS if k in b}, task="convert to 4-bit (once)", prompt_id=b["part"],
                run=0, note=f"{b['size_mb']} MB on disk, {b['on_card_mb']} MB on the card")
    row({k: load.get(k) for k in FIELDS if k in load}, task="cold load", prompt_id="", run=0,
        note="pipeline and transformer from the 4-bit cache onto the card")
    log(f"cold load {load['load_s']} s, card {load['vram_peak_mb']} MB, RAM peak {load['ram_peak_mb']} MB")

    edit_extra = json.loads(a.edit_extra)

    def draw(task, pid, prompt, w, h, run, image=None, name=None):
        picture, stats = engine.draw(prompt, w, h, seed=SEED, image=image, **(edit_extra if image is not None else {}))
        file = ""
        if picture is not None:
            tag = "-" + a.note.replace(" ", "_")[:24] if a.note else ""
            file = name or f"{a.model}-{task.replace(' ', '_')}-{pid}-{w}x{h}{tag}.png"
            picture.save(images / file)
        row(stats, task=task, prompt_id=pid, run=run, image=file)
        log(f"{task} {pid} {w}x{h}: {stats['seconds']} s (text {stats['swap_to_text_encoder_s']}+{stats['text_encoder_s']}, "
            f"swap {stats['swap_to_transformer_s']}, draw {stats['draw_s']}), card {stats['vram_start_mb']}->{stats['vram_peak_mb']} MB, "
            f"shared +{stats['shared_growth_mb']} MB, RAM {stats['ram_start_mb']}->{stats['ram_peak_mb']} MB {stats['error']}")
        return picture, stats

    # the first picture after a load pays one-off costs (CUDA start-up): recorded, not part of the medians
    draw("first picture after load", "fox", PROMPTS["fox"], 512, 512, 0, name=f"{a.model}-first.png")

    for w, h in a.sizes:
        for run, (pid, prompt) in enumerate(PROMPTS.items(), 1):
            _, stats = draw("text to image", pid, prompt, w, h, run)
            if stats["error"]:
                break
        else:
            # the same prompt again: the text encoder is skipped, the transformer is already on the card
            draw("text to image, same prompt again", "fisherman", PROMPTS["fisherman"], w, h, 4,
                 name=f"{a.model}-repeat-{w}x{h}.png")

    if spec.can_edit and not a.no_edit:
        from PIL import Image

        source = Path(a.edit_input) if a.edit_input else images / f"{a.model}-text_to_image-fox-1024x1024.png"
        if not source.exists():
            log(f"no picture to edit at {source}: skipping edits")
        else:
            picture = Image.open(source).convert("RGB")
            for w, h in a.edit_sizes:
                for run, (pid, prompt) in enumerate(EDITS.items(), 1):
                    draw("edit", pid, prompt, w, h, run, image=picture)

    engine.unload()
    f.close()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
