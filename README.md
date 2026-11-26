# 8 GB Image Lab

Run current open image models on an ordinary gaming PC: an NVIDIA card with **8 GB of VRAM**
and **16 GB of RAM**, Windows 11. One command sets it up and opens a page on your own PC
where you type a prompt, press Generate, and watch the graphics memory, the system RAM and
the clock while it draws.

> **What has been tested (2026-11-26):** Qwen-Image 2.1 Turbo, new pictures and edits, on one
> PC (below). The MOSS-Image and FLUX.2 klein options are in the menu but **have not been run
> yet**. Expect problems with them, and please open an issue if you try them.

It was built for a Below the Headline video that tests the claim "you need a 24 GB card for
modern image models". The numbers below were measured on one RTX 3070. Yours will differ.

<!-- NUMBERS:START -->
## Measured numbers

Qwen-Image 2.1 Turbo (8 steps), RTX 3070 8 GB, 16 GB DDR4-3200, Ryzen 5 5600X, Windows 11,
driver 621.24. Other programs closed; 1.1 GB of graphics memory in use before the app started.
25 November 2026. "Time" is the app's "Done in" figure.

| Run | Time | Graphics memory peak |
|---|---|---|
| Loading the model (first picture after starting the app) | 31–34 s, before the picture | — |
| New picture, 1024 x 1024, new prompt (7 prompts) | 33–38 s | 6.3–6.9 GB |
| Same prompt, new seed, 1024 x 1024 (the prompt is not read again) | 18 s | 6.3 GB |
| Same prompt, 512 x 512 / 768 x 768 | 4.6 s / 9.9 s | 6.4–6.5 GB |
| Same prompt, 1344 x 768 / 768 x 1344 / 1344 x 1344 | 18 s / 18 s / 33 s | 6.7–6.8 GB |
| Edit, 768 x 768 (3 edits) | 134–138 s | 8.1 GB, small spill into system RAM |
| Edit, 1024 x 1024 (3 edits) | 207–209 s | 8.0 GB |

All new pictures and the 768 edits came out right. At 1024 one edit (changing the text on a
sign) did nothing and the picture came out grainy, so edit at 768.

**Other programs on the card matter a lot.** The same tests with another program holding
3.7 GB of graphics memory in the background: new pictures 41–44 s, edits at 768 199–215 s,
edits at 1024 344–355 s. Check "Dedicated GPU memory" in Task Manager before you start; it
should be under about 2 GB.

**RAM is the tight part.** Making the 4-bit copies (once) needs most of the 16 GB. In the
first test, with a browser and other apps open, system RAM reached 100% for under a minute
while the text encoder was converted. Nothing crashed. Close what you can for the first run.
<!-- NUMBERS:END -->

## What you need

- Windows 10 or 11, an NVIDIA card with 8 GB of VRAM or more, a current NVIDIA driver.
- 16 GB of RAM. This is the tight part, tighter than the 8 GB of VRAM. Close what you can:
  with a browser and a few apps open the test PC had about 5 GB of RAM free.
- [Python 3.12](https://www.python.org/downloads/) (tick "Add to PATH" in the installer).
- Disk space and patience for the first run. Each model is downloaded at full size from
  Hugging Face and converted once to a small 4-bit copy:

  | Model | Download | 4-bit copy kept | Licence |
  |---|---|---|---|
  | Qwen-Image 2.1 Turbo | about 32 GB | about 10 GB | Qwen Research License: research and evaluation only, **not commercial** |
  | Qwen-Image 2.1 | about 32 GB (18 GB shared with Turbo) | about 10 GB (5 GB shared) | same |
  | MOSS-Image SFT | about 26 GB | about 9 GB | Apache-2.0 |
  | FLUX.2 klein 4B | about 15 GB | about 5 GB | Apache-2.0 |

  Only the model you pick is downloaded. The full-size files stay in the Hugging Face cache
  (`%USERPROFILE%\.cache\huggingface`); delete them there once the 4-bit copy exists if you
  need the space. The 4-bit copies are in `%USERPROFILE%\.cache\8gb-image-lab`.

## Run it

```
git clone https://github.com/Rusya13/8gb-image-lab
cd 8gb-image-lab
.\run.bat
```

Or open the folder in Explorer and double-click `run.bat`. (In PowerShell the `.\` in front is needed.)

Clone it into a short folder such as `C:\demo`. Some of PyTorch's files sit deep inside
`.venv`, and in a long folder path the install fails with "No such file or directory"
(Windows' 260-character path limit).

`run.bat` makes a private Python environment in `.venv`, installs the exact package versions
in `requirements.lock` (about 3 GB), and opens http://127.0.0.1:8188. Pick a model and press
Generate. The first picture with a model takes a long time: the download, then a few minutes
of converting. After that a model loads in seconds.

To edit a picture, choose a file under "Picture to edit" and describe the change.

## How it fits in 8 GB and 16 GB

1. **4-bit weights.** The transformer (the part that draws) and the text encoder (the part
   that reads your prompt) are converted to 4-bit with bitsandbytes NF4. Qwen-Image's text
   encoder goes from 16.7 GB to 5.2 GB.
2. **One big part at a time, on the card.** The text encoder and the transformer do not fit
   on 8 GB together, so the program loads the encoder, reads the prompt, drops it, and loads
   the transformer. The part that is not in use stays on disk, not in RAM. The same prompt
   with a new seed skips the encoder.
3. **A limit on its own RAM.** Reading a 13 GB model file maps it into memory. The program
   caps its own working set (between 1.2 and 2.2 GB on a 16 GB PC, depending on how much RAM is free at the start) so those pages are handed back as they
   pass through, and it writes its converted files in small pieces, flushing each one.
4. **The VAE decodes in tiles.**

What happens when it does not fit: on Windows the NVIDIA driver does not stop with "out of
memory". It moves the overflow into system RAM ("shared GPU memory") and everything runs many
times slower. The page shows this as "Spilled into system RAM". If that number grows while a
picture is drawing, pick a smaller size or close other programs that use the card.

## Measure your own PC

```
bench.bat qwen-turbo
bench.bat moss
bench.bat klein
```

Each writes one row per picture to `results\results.csv` (seconds per stage, card memory
before and at the peak, spill into RAM, system RAM before and at the peak) and saves the
pictures in `results\images`.

## Things to know

- It listens on `127.0.0.1` only. Nothing is sent anywhere except the model download from
  huggingface.co.
- MOSS-Image is run with its authors' own pipeline file, downloaded from their model
  repository at a pinned revision. That is code from the model's authors running on your PC.
- Model revisions are pinned in `app/engine.py`, so you get the same files that were measured.
- 4-bit weights are not identical to the full-precision model. Pictures can differ.
- Check each model's licence before you use its pictures for anything commercial.
- Tested on one PC (RTX 3070, driver 621.24, Windows 11, Python 3.12). No AMD or Intel cards,
  no Linux or macOS.

## Licence

The code in this repository is MIT (see `LICENSE`). The models are not part of it and have
their own licences, listed above.
