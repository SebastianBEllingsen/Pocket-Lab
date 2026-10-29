#!/bin/bash
# Experiment 3, unattended: wait for generation -> merge shards -> three runs (C, D, E) -> score on the 6 VALIDATION photos.
cd ~/synth-detector || exit 1
H=$HOME/synth-detector
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "waiting for generation"
while pgrep -f "[c]ellcount.synth --n" >/dev/null; do sleep 10; done
log "generation finished; merging shards"
mkdir -p synth_v2/images/train synth_v2/images/val synth_v2/labels/train synth_v2/labels/val
for k in 0 1 2 3 4 5 6 7; do
  mv v2_parts/p$k/images/train/* synth_v2/images/train/ && mv v2_parts/p$k/labels/train/* synth_v2/labels/train/ || { log "MERGE FAILED p$k"; exit 1; }
done
mv v2_parts/pval/images/val/* synth_v2/images/val/ && mv v2_parts/pval/labels/val/* synth_v2/labels/val/ || { log "MERGE FAILED val"; exit 1; }
head -1 v2_parts/p0/manifest.csv > synth_v2/manifest.csv
for d in v2_parts/p?/ v2_parts/pval/; do tail -n +2 $d/manifest.csv >> synth_v2/manifest.csv; done
head -1 v2_parts/p0/wbc_types.csv > synth_v2/wbc_types.csv
for d in v2_parts/p?/ v2_parts/pval/; do tail -n +2 $d/wbc_types.csv >> synth_v2/wbc_types.csv; done
log "merged: $(ls synth_v2/images/train | wc -l) train, $(ls synth_v2/images/val | wc -l) val scenes"
ls -d $H/synth_v2/images/train/*.jpg | sort | head -1440 > synth_v2_1440.txt
ls -d $H/synth_out/images/train/*.jpg | sort | head -1440 > synth_v1_1440.txt
mk() { # name, extra train entries...
  n=$1; shift
  { echo "train:"; echo "  - $H/merged/images/train"; for e in "$@"; do echo "  - $e"; done
    echo "val: $H/merged/images/val"; echo "test: $H/merged/images/test"
    printf "names:\n  0: rbc\n  1: wbc\n  2: platelet\n  3: smudge\n"; } > $n.yaml
}
mk C_public_synthv2_all $H/synth_v2/images/train
mk D_public_synthv2_1440 $H/synth_v2_1440.txt
mk E_public_synthv1_1440 $H/synth_v1_1440.txt
cd detector
for r in C_public_synthv2_all:C D_public_synthv2_1440:D E_public_synthv1_1440:E; do
  y=${r%%:*}; n=${r##*:}
  log "start run_$n ($y)"
  nohup ../.venv/bin/python train_yolo26.py --data $H/$y.yaml --model yolo26s.pt --epochs 100 --batch 16 --device 0 --name run_$n --eval-split val > ../train_run_$n.log 2>&1 &
done
sleep 60
while pgrep -f "[t]rain_yolo26.py" >/dev/null; do sleep 20; done
log "training finished; scoring on the 6 validation photos"
../.venv/bin/python ../val_photos.py A=runs/detect/run_A/weights/best.pt B=runs/detect/run_B/weights/best.pt \
  C=runs/detect/run_C/weights/best.pt D=runs/detect/run_D/weights/best.pt E=runs/detect/run_E/weights/best.pt > ../results_exp3_val_photos.txt 2>&1
../.venv/bin/python ../ann_multi.py unk_40x_02 A B C D E > ../ann_multi.log 2>&1
for n in A B C D E; do echo "== run $n (merged val)"; tail -c 3500 ../train_run_$n.log | tr "\r" "\n" | grep -E "val mAP|per-class" ; done > ../results_exp3_public_val.txt
log "ALL DONE: results_exp3_val_photos.txt, results_exp3_public_val.txt, ann_*/unk_40x_02_yolo.jpg"
