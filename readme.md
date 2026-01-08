# Experiment Runner v2 (Multi-model + Model-specific config + LR scheduler)

จุดที่เพิ่ม:
- รองรับโมเดลเพิ่มได้ง่าย (แค่สร้างไฟล์ใน `src/models/` แล้ว register)
- config รองรับ optimizer/scheduler เฉพาะโมเดล (เช่น DnCNN ใช้ SGD + MultiStepLR)
- Text log + metrics.jsonl + checkpoints แยกเป็น run ต่อโมเดลอัตโนมัติ

## Run
```bash
python run.py --config configs/unet_rician.json
python run.py --config configs/ucx_rician.json
python run.py --config configs/dncnn_rician.json
```

## DnCNN LR schedule (ตามที่ต้องการ)
ตั้งใน config:
```json
"scheduler_cfg": {"name":"multistep","args":{"milestones":[10,20],"gamma":0.1}}
```
ผลคือ:
- epochs 1-10: lr = 0.001
- epochs 11-20: lr = 0.0001
- epochs 21+:   lr = 0.00001

## Override ให้แก้ง่าย (ไม่ต้องแก้ไฟล์)
```bash

Dncnn:
python run.py --config configs/dncnn_rician.json --override training.epochs=40 training.optimizer_cfg.args.lr=0.0005

GAN:
python run.py --config configs/gan_rician.json --override training.epochs=40 training.optimizer_G_cfg.args.lr=0.0005 training.optimizer_D_cfg.args.lr=0.0005 
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