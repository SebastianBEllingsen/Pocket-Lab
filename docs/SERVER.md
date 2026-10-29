# Training server (uia Coder workspace)

Set up 2026-10-29 for detector experiments on synthetic data. The local GPU (Quadro T2000, 4 GB) is too small for bigger runs.

## Access
- Needs the university network or VPN: `coder.internal.uia.no` does not resolve otherwise.
- Connect: `coder ssh fuckthisshit` (workspace name). One-off command: `coder ssh fuckthisshit -- '<command>'`.
- Hardware: Tesla V100-SXM3 32 GB, about 977 GB free in `/home/coder`. Driver supports CUDA up to 12.9.
- No `tmux` or `python3` on the PATH; `uv` is at `~/.local/bin/uv`. Use `nohup ... &` for long runs (they survive a dropped connection).
- Only `~/synth-detector` belongs to this project. Other folders in the home directory belong to other uses and are left alone.

## Layout of `~/synth-detector`
```
.venv/             uv venv, Python 3.12, torch 2.14 (CUDA 12.6 build), ultralytics, opencv-python-headless, scikit-image, scipy
synth_out/         synthetic YOLO dataset (copy of Data_Sets/synth_out; data.yaml path edited for the server)
detector/          the teammate's branch (git archive of origin/cell-detection-v2), data/own_v2 = our phone photos
run_eval.py        runs the teammate's evaluate.py on chosen weights: python ../run_eval.py name=path/to/best.pt
box_sizes.py       compares predicted and hand-drawn red-cell box sizes
train_synth_only.log   log of the first run
```
Weights of a run land in `detector/runs/detect/<name>/weights/best.pt`.

## Gotchas found
- PyPI's newest PyTorch is built for CUDA 13.2 and fails on the server driver. Install the CUDA 12.6 build: `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
- Ultralytics pulls in the full `opencv-python`, which needs `libxcb`; replace it with `opencv-python-headless`.
- `coder ssh` prints a version-mismatch notice (client v2.35.4, server v2.37.4); harmless.

## Repeat the first experiment
Copy data (from the project root, on the local machine):
```bash
tar czf - -C Data_Sets synth_out | coder ssh fuckthisshit -- 'mkdir -p ~/synth-detector && tar xzf - -C ~/synth-detector'
git -C ../Course-Project-detector archive --format=tar HEAD | coder ssh fuckthisshit -- 'mkdir -p ~/synth-detector/detector && tar xf - -C ~/synth-detector/detector'
# then: sed -i "s#^path: .*#path: $HOME/synth-detector/synth_out#" ~/synth-detector/synth_out/data.yaml  (on the server)
```
Train (on the server, in `~/synth-detector/detector`):
```bash
nohup ../.venv/bin/python train_yolo26.py --data $HOME/synth-detector/synth_out/data.yaml --model yolo26n.pt \
  --epochs 40 --batch 16 --device 0 --name synth_only --eval-split val > ../train_synth_only.log 2>&1 &
```
Evaluate on the teammate's real test (48 hand-counted squares):
```bash
../.venv/bin/python ../run_eval.py synth=runs/detect/synth_only/weights/best.pt
```
The local copy of the teammate's branch is a git worktree: `git worktree add --detach ../Course-Project-detector origin/cell-detection-v2` (remove with `git worktree remove ../Course-Project-detector`).

## Rules for use
- Ask Sebastian before starting a new run, copying data or installing anything (he asked to be consulted).
- Nothing is pushed anywhere; results are written down in `docs/DECISIONS.md`.

## Unattended jobs (no tmux on the server)
- Long jobs are started with `nohup ... &`. They keep running when the laptop sleeps or the connection drops (checked 2026-10-29). Only steps driven interactively from the laptop need it online, so whole chains are written as one server-side script.
- Heavy work (data generation and training) runs on the server, not on the laptop (fans, battery). Generation is sharded: `python -m cellcount.synth --n 625 --seed 11 --start $((k*625)) --out ../v2_parts/p$k` for k = 0..7 gives the same scenes as one run (scene k uses seed `[seed, start+k]`). The code and bank live in `~/synth-detector/gen`.
- Experiment 3 script: `~/synth-detector/pipeline_exp3.sh` (copy in `docs/handover/pipeline_exp3.sh`). It waits for generation, merges shards into `synth_v2/`, trains runs C, D and E concurrently, scores A-E on the six validation photos and writes `results_exp3_val_photos.txt`, `results_exp3_public_val.txt` and `ann_*/unk_40x_02_yolo.jpg`. Progress: `cat ~/synth-detector/pipeline_exp3.log`.
- Check from anywhere on the university network: `coder ssh fuckthisshit -- 'cat ~/synth-detector/pipeline_exp3.log; tail -n 30 ~/synth-detector/results_exp3_val_photos.txt'`.
