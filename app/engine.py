"""Runs one image model at a time on an 8 GB NVIDIA card and a 16 GB PC.

How it fits:
- The big parts (the transformer that draws, the text encoder that reads the prompt) are
  converted once to 4-bit (bitsandbytes NF4) and saved under CACHE. Later starts read the
  small 4-bit files instead of the full-size download.
- Only one big part is in memory at a time, and it sits on the card, not in RAM. To read a
  new prompt the transformer is dropped and the text encoder is read from disk; then the
  other way round to draw. That costs a few seconds per new prompt and keeps system RAM
  free. (Parking the idle part in RAM, as diffusers' model offload does, needs about 10 GB
  of RAM for Qwen-Image: too much beside a browser on a 16 GB PC.)
- The VAE decodes in tiles.

Nothing here is specific to one PC. Model files come from the Hugging Face cache.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# The Xet download backend can stall on some connections; plain HTTP is slower but finishes.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

from . import monitor
from .degrid import degrid

CACHE = Path(os.environ.get("EIGHTGB_CACHE", Path.home() / ".cache" / "8gb-image-lab"))
BF16 = torch.bfloat16
CARD = "cuda"


@dataclass
class ModelSpec:
    key: str
    title: str
    repo: str
    revision: str          # the exact files the published numbers were measured with
    family: str            # "qwen" | "moss" | "klein"
    steps: int
    guidance: float        # 1.0 = no classifier-free guidance (one transformer pass per step)
    can_edit: bool
    licence: str
    download_gb: float     # what the first run downloads for this model
    shared_repo: str = ""  # where the text encoder and VAE come from, when not `repo`
    shared_revision: str = ""
    passes_per_step: int = 1


MODELS = {m.key: m for m in [
    ModelSpec("qwen-turbo", "Qwen-Image 2.1 Turbo (8 steps)", "Qwen/Qwen-Image-2.1-Turbo",
              "fc13ba1986fcb905854dc8de2f52eee395f49b5b", "qwen", steps=8, guidance=1.0, can_edit=True,
              licence="Qwen Research License (research and evaluation only, not commercial)", download_gb=31.6,
              shared_repo="Qwen/Qwen-Image-2.1", shared_revision="d26bb61231c349cf6b7896fa83353113880e1ba3"),
    ModelSpec("qwen", "Qwen-Image 2.1 (40 steps)", "Qwen/Qwen-Image-2.1",
              "d26bb61231c349cf6b7896fa83353113880e1ba3", "qwen", steps=40, guidance=4.0, can_edit=True,
              licence="Qwen Research License (research and evaluation only, not commercial)", download_gb=31.6,
              shared_repo="Qwen/Qwen-Image-2.1", shared_revision="d26bb61231c349cf6b7896fa83353113880e1ba3",
              passes_per_step=2),
    ModelSpec("moss", "MOSS-Image SFT (50 steps)", "OpenMOSS-Team/MOSS-Image-SFT",
              "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9", "moss", steps=50, guidance=3.0, can_edit=False,
              licence="Apache-2.0", download_gb=25.9, passes_per_step=2),
    ModelSpec("klein", "FLUX.2 klein 4B (4 steps)", "black-forest-labs/FLUX.2-klein-4B",
              "e7b7dc27f91deacad38e78976d1f2b499d76a294", "klein", steps=4, guidance=1.0, can_edit=True,
              licence="Apache-2.0", download_gb=15.2),
]}


def versions() -> dict:
    import accelerate
    import bitsandbytes
    import diffusers
    import transformers

    return {"torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__,
            "accelerate": accelerate.__version__, "gpu": monitor.gpu_name(), "driver": monitor.driver_version()}


def cap_own_ram(mb: int | None = None) -> int:
    """Limits how much system RAM this process may hold at once (Windows working-set cap).

    Reading a 5 to 13 GB model file maps it into the process; without a cap those pages pile
    up until RAM is full, even though they are only passing through on their way to the card.
    With the cap, Windows hands the oldest pages straight back. Returns the cap in MB, or 0
    when none was set. Default on a PC with up to 17 GB of RAM: about a third of the RAM that
    is free when the program starts, between 1,200 and 2,200 MB; none on a bigger PC.
    Override with EIGHTGB_RAM_CAP_MB (0 turns it off).
    """
    if sys.platform != "win32":
        return 0
    import ctypes

    import psutil

    if mb is None:
        env = os.environ.get("EIGHTGB_RAM_CAP_MB")
        v = psutil.virtual_memory()
        small_pc = v.total <= 17 * 2**30
        mb = int(env) if env is not None else (min(2200, max(1200, int(0.35 * v.available / 2**20))) if small_pc else 0)
    if mb <= 0:
        return 0
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessWorkingSetSizeEx.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
    hard_max, no_hard_min = 0x4, 0x2
    ok = k.SetProcessWorkingSetSizeEx(k.GetCurrentProcess(), 64 * 2**20, mb * 2**20, hard_max | no_hard_min)
    return mb if ok else 0


RAM_CAP_MB = cap_own_ram()


class _FlushedSaves:
    """While active, every model file written is flushed to disk before the next one starts.
    Otherwise Windows keeps gigabytes of not-yet-written data in RAM during a save."""

    def __enter__(self):
        import safetensors.torch
        import transformers.modeling_utils as tm

        self._original = safetensors.torch.save_file

        def save_file(tensors, filename, *args, **kwargs):
            self._original(tensors, filename, *args, **kwargs)
            with open(filename, "rb+") as f:
                os.fsync(f.fileno())

        self._tm, self._tm_original = tm, tm.safe_save_file
        safetensors.torch.save_file = save_file   # diffusers looks it up here on each call
        tm.safe_save_file = save_file             # transformers imported it by name
        return self

    def __exit__(self, *exc):
        import safetensors.torch

        safetensors.torch.save_file = self._original
        self._tm.safe_save_file = self._tm_original
        return False


def free_card() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nf4(skip: list[str] | None = None, lib: str = "diffusers"):
    """4-bit NF4 settings. `skip` names Linear layers that stay in bf16."""
    if lib == "transformers":
        from transformers import BitsAndBytesConfig
    else:
        from diffusers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=BF16,
                              llm_int8_skip_modules=skip)


def _folder_mb(path: Path) -> int:
    return int(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20)


def build_part(name: str, load, log=print) -> dict:
    """Converts one part to 4-bit and saves it under CACHE/name, if it is not there yet.

    `load()` returns the quantised model (loaded from the full-size files). Returns what the
    conversion cost: seconds, peak card memory, peak system RAM, the saved size.
    """
    out = CACHE / name
    if (out / "config.json").exists():
        return {"part": name, "built": False, "size_mb": _folder_mb(out)}
    log(f"[build] {name}: converting to 4-bit (once) ...")
    tmp = CACHE / (name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    with monitor.Peak() as peak:
        model = load()
        gc.collect()
        on_card_mb = int(torch.cuda.memory_allocated() / 2**20)
        # Written in small pieces, each flushed: one 5 GB buffer in RAM beside the copy on the
        # card is what took a 16 GB PC to 100% RAM.
        with _FlushedSaves():
            model.save_pretrained(tmp, max_shard_size="250MB")
        del model
        free_card()
    os.replace(tmp, out)
    result = {"part": name, "built": True, "size_mb": _folder_mb(out), "on_card_mb": on_card_mb, **peak.result}
    log(f"[build] {name}: {result['seconds']} s, {result['size_mb']} MB on disk, {on_card_mb} MB on the card, "
        f"card peak {result['vram_peak_mb']} MB, RAM peak {result['ram_peak_mb']} MB")
    return result


class Family:
    """One model family: how its parts are converted, loaded, and called."""

    text_encoder_cache = ""  # the 4-bit text encoder's folder under CACHE

    def __init__(self, spec: ModelSpec, log=print):
        self.spec = spec
        self.log = log
        self.pipe = None

    @property
    def transformer_cache(self) -> str:
        return f"{self.spec.key}-transformer-nf4"

    def build(self) -> list[dict]:
        return [build_part(self.text_encoder_cache, self._quantise_text_encoder, self.log),
                build_part(self.transformer_cache, self._quantise_transformer, self.log)]

    def text_encoder(self):
        """The 4-bit text encoder, read from CACHE straight onto the card."""
        raise NotImplementedError

    def transformer(self):
        raise NotImplementedError

    def make_pipe(self):
        """The pipeline without its two big parts (they are attached when needed)."""
        raise NotImplementedError

    def encode(self, prompt: str, image):
        """Runs the text encoder (already attached). Returns what `call` needs, on the CPU."""
        raise NotImplementedError

    def call(self, embeds, width, height, steps, generator, image):
        raise NotImplementedError


# --- Qwen-Image 2.1 (base and Turbo) ----------------------------------------------------------

class Qwen(Family):
    text_encoder_cache = "qwen-image-2.1-text-encoder-nf4"

    class Headless(torch.nn.Module):
        """The text encoder without its output head. The pipeline reads hidden states, never
        logits, so the 1.2 GB head is dead weight; the pipeline only needs `.model` and a call."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, **kwargs):
            return self.model(**kwargs)

    def _quantise_text_encoder(self):
        from transformers import Qwen3VLModel  # the model without the language-model head

        s = self.spec
        return Qwen3VLModel.from_pretrained(
            s.shared_repo, revision=s.shared_revision, subfolder="text_encoder",
            quantization_config=_nf4(lib="transformers"), dtype=BF16, device_map={"": 0})

    # Qwen-Image 2.1's own VAE leaves a grid over every picture: a 2-pixel checker and a pattern
    # that repeats every 16 pixels (one latent pixel), seen on skin, fur and smooth areas. It is
    # the same with full-precision weights, so it is not caused by the 4-bit copies.
    #  - The decoder is swapped for madebyollin's texture-fix finetune of the same VAE (same
    #    latents, same licence), which removes the 2-pixel checker and part of the 16-pixel grid.
    #  - app/degrid.py then removes most of what is left of the 16-pixel grid (about 2 s).
    VAE_REPO = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
    VAE_REVISION = "9a10df811f183323e4e94d255c18add1924ce340"
    DEGRID = True

    def _quantise_transformer(self):
        from diffusers import QwenImage21Transformer2DModel

        return QwenImage21Transformer2DModel.from_pretrained(
            self.spec.repo, revision=self.spec.revision, subfolder="transformer",
            quantization_config=_nf4(), torch_dtype=BF16)

    def text_encoder(self):
        from transformers import Qwen3VLModel

        model = Qwen3VLModel.from_pretrained(CACHE / self.text_encoder_cache, dtype=BF16, device_map={"": 0})
        return self.Headless(model).eval()

    def transformer(self):
        from diffusers import QwenImage21Transformer2DModel

        return QwenImage21Transformer2DModel.from_pretrained(
            CACHE / self.transformer_cache, torch_dtype=BF16, device_map=CARD).eval()

    def make_pipe(self):
        from diffusers import QwenImage21Pipeline

        s = self.spec
        return QwenImage21Pipeline.from_pretrained(s.repo, revision=s.revision, text_encoder=None,
                                                   transformer=None, vae=self.vae(), torch_dtype=BF16)

    def vae(self):
        from diffusers import AutoencoderKLQwenImage21

        vae = AutoencoderKLQwenImage21.from_pretrained(
            self.VAE_REPO, revision=self.VAE_REVISION, torch_dtype=BF16).to(CARD)
        vae.enable_tiling()
        return vae

    def _condition_images(self, image, resolution: int):
        """The reference picture resized the way the pipeline does before the text encoder sees it."""
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        if image.mode != "RGBA":
            image = image.convert("RGBA")
        w, h, _ = calculate_dimensions(resolution * resolution, image.width / image.height)
        return [self.pipe.image_processor.resize(image, width=w, height=h)]

    def encode(self, prompt: str, image, resolution: int = 1024, **_):
        pipe = self.pipe
        images = self._condition_images(image, resolution) if image is not None else None
        out = {"cond": tuple(None if t is None else t.cpu() for t in
                             pipe.encode_prompt(prompt=prompt, image=images, device=torch.device(CARD)))}
        if self.spec.guidance > 1:
            out["neg"] = tuple(None if t is None else t.cpu() for t in
                               pipe.encode_prompt(prompt=" ", image=images, device=torch.device(CARD)))
        return out

    def call(self, embeds, width, height, steps, generator, image, resolution: int = 1024, **extra):
        pipe = self.pipe
        queue = [embeds["cond"]] + ([embeds["neg"]] if "neg" in embeds else [])

        def cached(**_):
            # The pipeline asks for the prompt first, then (with guidance) the negative prompt.
            return tuple(None if t is None else t.to(CARD) for t in queue.pop(0))

        pipe.encode_prompt = cached  # the text encoder is not in memory during the steps
        try:
            kwargs = dict(prompt="", width=width, height=height, generator=generator,
                          output_resolution=resolution, **extra)
            if image is not None:
                kwargs["image"] = image
            if self.spec.guidance > 1:
                kwargs.update(num_inference_steps=steps, true_cfg_scale=self.spec.guidance, negative_prompt=" ")
            return pipe(**kwargs).images[0]
        finally:
            del pipe.encode_prompt


# --- MOSS-Image ----------------------------------------------------------------------------

class Moss(Family):
    text_encoder_cache = "moss-image-text-encoder-nf4"
    # Only the 42 blocks' projections go to 4-bit. The small input/output layers stay in bf16,
    # and so does each block's "q" projection: the authors' block reads its working precision
    # from that layer's weight, which would be the packed 4-bit storage type otherwise.
    KEEP_BF16 = ["video_embedder", "text_embedder", "time_embedding", "shared_block_modulation",
                 "final_linear_video", "video_head_modulation", "q"]

    def _module(self):
        """The authors' pipeline file, from the model repository at the pinned revision.
        It is code from the model's authors (Apache-2.0) and it runs on your PC."""
        name = "pipeline_moss_image"
        if name not in sys.modules:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.spec.repo, "pipeline_moss_image.py", revision=self.spec.revision)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def _quantise_text_encoder(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.spec.repo, revision=self.spec.revision, subfolder="text_encoder",
                                         quantization_config=_nf4(lib="transformers"), dtype=BF16,
                                         device_map={"": 0})

    def _quantise_transformer(self):
        return self._module().MOSSImageTransformer.from_pretrained(
            self.spec.repo, revision=self.spec.revision, subfolder="transformer",
            quantization_config=_nf4(self.KEEP_BF16), torch_dtype=BF16)

    def text_encoder(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(CACHE / self.text_encoder_cache, dtype=BF16, device_map={"": 0}).eval()

    def transformer(self):
        return self._module().MOSSImageTransformer.from_pretrained(
            CACHE / self.transformer_cache, torch_dtype=BF16, device_map=CARD).eval()

    def make_pipe(self):
        from transformers import AutoTokenizer

        s, m = self.spec, self._module()
        vae = self.vae()
        tokenizer = AutoTokenizer.from_pretrained(s.repo, revision=s.revision, subfolder="tokenizer")
        scheduler = m.MOSSImageScheduler.from_config(
            m.MOSSImageScheduler.load_config(s.repo, revision=s.revision, subfolder="scheduler"))
        return m.MOSSImagePipeline(transformer=None, scheduler=scheduler, text_encoder=None,
                                   tokenizer=tokenizer, vae=vae)

    def vae(self):
        from diffusers import AutoencoderKLFlux2

        s = self.spec
        vae = AutoencoderKLFlux2.from_pretrained(s.repo, revision=s.revision, subfolder="vae",
                                                 torch_dtype=torch.float32).to(CARD).eval()
        vae.enable_tiling()
        return vae

    def encode(self, prompt: str, image, **_):
        pos, pos_mask = self.pipe.encode_prompt(prompt)
        neg, neg_mask = self.pipe.encode_prompt("")
        return {"pos": pos.cpu(), "pos_mask": pos_mask.cpu(), "neg": neg.cpu(), "neg_mask": neg_mask.cpu()}

    def call(self, embeds, width, height, steps, generator, image, **extra):
        e = {k: v.to(CARD) for k, v in embeds.items()}
        return self.pipe(prompt_embeds=e["pos"], prompt_attention_mask=e["pos_mask"],
                         negative_prompt_embeds=e["neg"], negative_prompt_attention_mask=e["neg_mask"],
                         width=width, height=height, num_inference_steps=steps,
                         guidance_scale=self.spec.guidance, uncond_skip_layers=[22],
                         generator=generator, **extra).images[0]


# --- FLUX.2 klein 4B -----------------------------------------------------------------------

class Klein(Family):
    text_encoder_cache = "flux2-klein-4b-text-encoder-nf4"

    def _quantise_text_encoder(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.spec.repo, revision=self.spec.revision, subfolder="text_encoder",
            quantization_config=_nf4(lib="transformers"), dtype=BF16, device_map={"": 0})

    def _quantise_transformer(self):
        from diffusers import Flux2Transformer2DModel

        return Flux2Transformer2DModel.from_pretrained(
            self.spec.repo, revision=self.spec.revision, subfolder="transformer",
            quantization_config=_nf4(), torch_dtype=BF16)

    def text_encoder(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(CACHE / self.text_encoder_cache, dtype=BF16,
                                                    device_map={"": 0}).eval()

    def transformer(self):
        from diffusers import Flux2Transformer2DModel

        return Flux2Transformer2DModel.from_pretrained(CACHE / self.transformer_cache, torch_dtype=BF16,
                                                       device_map=CARD).eval()

    def make_pipe(self):
        from diffusers import Flux2KleinPipeline

        s = self.spec
        return Flux2KleinPipeline.from_pretrained(s.repo, revision=s.revision, text_encoder=None,
                                                  transformer=None, vae=self.vae(), torch_dtype=BF16)

    def vae(self):
        from diffusers import AutoencoderKLFlux2

        s = self.spec
        vae = AutoencoderKLFlux2.from_pretrained(s.repo, revision=s.revision, subfolder="vae",
                                                 torch_dtype=BF16).to(CARD)
        vae.enable_tiling()
        return vae

    def encode(self, prompt: str, image, **_):
        embeds, _ids = self.pipe.encode_prompt(prompt=prompt, device=torch.device(CARD))
        return {"embeds": embeds.cpu()}

    def call(self, embeds, width, height, steps, generator, image, **extra):
        kwargs = dict(prompt_embeds=embeds["embeds"].to(CARD), width=width, height=height,
                      num_inference_steps=steps, guidance_scale=1.0, generator=generator, **extra)
        if image is not None:
            kwargs["image"] = image
        return self.pipe(**kwargs).images[0]


FAMILIES = {"qwen": Qwen, "moss": Moss, "klein": Klein}


# --- the engine ------------------------------------------------------------------------------

class _OnCard:
    """Mixed into the pipeline class: tells it where to compute while a big part is absent."""

    @property
    def _execution_device(self):
        return torch.device(CARD)


def _attach(pipe, name: str, part) -> None:
    """Sets a pipeline part directly, without the pipeline's own registration."""
    object.__setattr__(pipe, name, part)


def _image_key(image) -> str:
    if image is None:
        return ""
    return hashlib.sha1(image.tobytes()).hexdigest() + f"{image.size}"


class Engine:
    """Holds one model. `load(key)` then `draw(...)`; every call returns what it cost."""

    def __init__(self, log=print):
        self.log = log
        self.family: Family | None = None
        self.key = None
        self.lock = threading.Lock()
        self.progress = {"state": "idle", "step": 0, "steps": 0, "model": None}
        self._on_card = None          # "text_encoder" | "transformer" | None
        self._embeds = (None, None)   # (key, embeddings) of the last prompt
        self._passes = 0

    @property
    def spec(self) -> ModelSpec | None:
        return self.family.spec if self.family else None

    def _set(self, **kw):
        self.progress = {**self.progress, **kw}

    def unload(self) -> None:
        if self.family is not None and self.family.pipe is not None:
            _attach(self.family.pipe, "text_encoder", None)
            _attach(self.family.pipe, "transformer", None)
        self.family = None
        self.key = None
        self._on_card = None
        self._embeds = (None, None)
        free_card()
        self.progress = {"state": "idle", "step": 0, "steps": 0, "model": None}

    def _want(self, part: str) -> float:
        """Puts `part` on the card (dropping the other big part first). Returns the seconds it took."""
        if self._on_card == part:
            return 0.0
        t0 = time.perf_counter()
        pipe = self.family.pipe
        _attach(pipe, "text_encoder", None)
        _attach(pipe, "transformer", None)
        self._on_card = None
        free_card()
        if part == "text_encoder":
            # The VAE goes too: every gigabyte on the card is also a gigabyte of Windows' commit
            # charge, and a 16 GB PC with apps open has little of that to spare.
            _attach(pipe, "vae", None)
            free_card()
            _attach(pipe, "text_encoder", self.family.text_encoder())
        else:
            if getattr(pipe, "vae", None) is None:
                _attach(pipe, "vae", self.family.vae())
            transformer = self.family.transformer()
            transformer.register_forward_hook(self._after_pass)
            _attach(pipe, "transformer", transformer)
        self._on_card = part
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _after_pass(self, *_):
        self._passes += 1
        self._set(step=self._passes // max(1, self.spec.passes_per_step))

    def load(self, key: str) -> dict:
        """Cold load: from nothing in memory to ready to draw (the transformer on the card).
        The first time, this also converts the model to 4-bit, which takes minutes."""
        with self.lock:
            self.unload()
            spec = MODELS[key]
            self.progress = {"state": "loading", "step": 0, "steps": 0, "model": key}
            family = FAMILIES[spec.family](spec, self.log)
            builds = family.build()
            with monitor.Peak() as peak:
                pipe = family.make_pipe()
                pipe.__class__ = type("OnCard" + type(pipe).__name__, (_OnCard, type(pipe)), {})
                pipe.set_progress_bar_config(disable=True)
                family.pipe = pipe
                self.family, self.key = family, key
                self._want("transformer")
            self._set(state="ready")
            return {"model": key, "load_s": peak.result["seconds"], "builds": builds, **peak.result}

    def draw(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0,
             steps: int | None = None, image=None, **extra) -> tuple:
        """Returns (PIL image or None, stats). `image` (a PIL image) turns the call into an edit."""
        if self.family is None:
            raise RuntimeError("load a model first")
        spec, family = self.spec, self.family
        if image is not None and not spec.can_edit:
            raise ValueError(f"{spec.title} cannot edit pictures")
        with self.lock:
            steps = steps or spec.steps
            self._passes = 0
            self._set(state="reading the prompt", step=0, steps=steps)
            torch.cuda.reset_peak_memory_stats()
            out, error = None, ""
            times = {"swap_to_text_encoder_s": 0.0, "text_encoder_s": 0.0, "swap_to_transformer_s": 0.0,
                     "draw_s": 0.0}
            with monitor.Peak() as peak:
                try:
                    with torch.inference_mode():
                        key = (prompt, _image_key(image), repr(sorted(extra.items())))
                        if self._embeds[0] != key:
                            times["swap_to_text_encoder_s"] = self._want("text_encoder")
                            t0 = time.perf_counter()
                            embeds = family.encode(prompt, image, **extra)
                            torch.cuda.synchronize()
                            times["text_encoder_s"] = time.perf_counter() - t0
                            self._embeds = (key, embeds)
                        times["swap_to_transformer_s"] = self._want("transformer")
                        self._set(state="drawing")
                        t0 = time.perf_counter()
                        out = family.call(self._embeds[1], width, height, steps,
                                          torch.Generator("cpu").manual_seed(seed), image, **extra)
                        torch.cuda.synchronize()
                        times["draw_s"] = time.perf_counter() - t0
                    if out is not None and getattr(family, "DEGRID", False):
                        # Qwen-Image 2.1's VAE leaves a faint 16-pixel grid; see app/degrid.py.
                        self._set(state="removing the VAE grid")
                        t0 = time.perf_counter()
                        out = degrid(out)
                        times["degrid_s"] = time.perf_counter() - t0
                except torch.OutOfMemoryError as e:
                    error = f"out of memory: {str(e)[:160]}"
                    free_card()
            stats = {
                "model": self.key, "width": width, "height": height, "steps": steps, "seed": seed,
                "edit": image is not None, "error": error,
                **{k: round(v, 2) for k, v in times.items()},
                "torch_peak_alloc_mb": int(torch.cuda.max_memory_allocated() / 2**20),
                "torch_peak_reserved_mb": int(torch.cuda.max_memory_reserved() / 2**20),
                **peak.result,
            }
            self._set(state="ready", step=steps if not error else 0)
            free_card()
            return out, stats
