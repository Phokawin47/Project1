# Runner Script

## Run
```bash
python run.py --config configs/unet.json
```
```bash
python run.py --config configs/dncnn_rician.json
```
```bash
python run.py --config configs/gan_rician.json
```
```bash
python run.py --config configs/armnet_rician.json
```

# Resume Train
ใช้ last.pt
```bash
python resume_training_stateful.py --run_dir <Parth of model (runs/dncnn/20260117_002549_dncnn_schedule_7e17edcc)> --ckpt last --max_minutes 175
```
ใช้ best.pt
```bash
python resume_training_stateful.py --run_dir <Parth of model (runs/dncnn/20260117_002549_dncnn_schedule_7e17edcc)> --ckpt best --max_minutes 175
```

## Dataset
This template uses your uploaded `BrainTumorDataset` (Rician noise) via `src/data/brain_tumor_user.py`.

...
## CC359 PreProcess

In the directory you must have file "./data/cc359_original/Original"
To prepare CC359 dataset with 288x288 shards, run:
```bash
python prepare_cc359_shards_288.py --zip Original.zip --extract_dir data\cc359_original --out_dir processed\cc359_shards_288 --crop_head --max_black_ratio 0.60 --black_thresh 0.02 --debug_jpg
```

To export sample image, run:
```bash
python prepare_cc359_shards_288.py --export_npz processed\cc359_shards_288\CC0001.npz --export_out_dir exports --export_fmt png
```

to export .npz file to image all run:
```bash
for %f in (processed\cc359_shards_288\*.npz) do (
  python prepare_cc359_shards_288.py --export_npz %f --export_out_dir exports_all --export_fmt png
)

```
