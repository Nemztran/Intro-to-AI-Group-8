# Huong dan train va danh gia

File nay huong dan cach train, test va xuat ket qua cho project Maze Solving voi SAC.

## 1. Chuan bi moi truong

Mo PowerShell tai thu muc repo:

```powershell
cd C:\Code\Project\Intro-to-AI-Group-8
```

Neu repo da co san `.venv`, dung Python trong virtual environment:

```powershell
.\.venv\Scripts\python.exe --version
```

Neu can cai lai dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiem tra nhanh code co import/chay duoc khong:

```powershell
.\.venv\Scripts\python.exe -m py_compile agent.py main.py model.py buffer.py gym_robotics_custom.py test.py _init_test.py evaluation.py evaluate_seeds.py export_tensorboard_metrics.py
.\.venv\Scripts\python.exe _init_test.py
```

## 2. Train SAC + Curiosity

Day la cau hinh chinh cua project: SAC co intrinsic curiosity reward.

```powershell
.\.venv\Scripts\python.exe main.py --checkpoint-dir checkpoints\sac_curiosity --experiment-name sac_curiosity --seed 0
```

Ket qua:

- Checkpoint luu trong `checkpoints\sac_curiosity`
- TensorBoard log luu trong `runs\...sac_curiosity...`

Mac dinh `main.py` train 2 phase:

- Phase 1: `STRAIGHT_MAZE`, 100 episodes, toi da 100 steps/episode
- Phase 2: `LARGE_MAZE`, 3000 episodes, toi da 500 steps/episode

Neu chi muon test nhanh pipeline:

```powershell
.\.venv\Scripts\python.exe main.py --phase1-episodes 5 --phase2-episodes 10 --checkpoint-dir checkpoints\sac_curiosity_smoke --experiment-name sac_curiosity_smoke --seed 0
```

## 3. Train SAC only

Dung cau hinh nay de so sanh ablation: SAC khong dung intrinsic curiosity.

```powershell
.\.venv\Scripts\python.exe main.py --disable-curiosity --checkpoint-dir checkpoints\sac_only --experiment-name sac_only --seed 0
```

Ket qua:

- Checkpoint luu trong `checkpoints\sac_only`
- TensorBoard log luu trong `runs\...sac_only...`

## 4. Test nhanh checkpoint

Mac dinh `test.py` load checkpoint trong `CHECKPOINT_DIR`.

Neu muon test checkpoint mac dinh `checkpoints`, chay:

```powershell
.\.venv\Scripts\python.exe test.py
```

Ket qua se co bang so sanh:

```text
Agent                Success Rate    Avg Steps   Avg Reward
------------------------------------------------------------
SAC Agent                 ...%          ...          ...
Random Agent              ...%          ...          ...
```

Neu muon test checkpoint o thu muc khac, sua hang `CHECKPOINT_DIR` trong `test.py`, vi du:

```python
CHECKPOINT_DIR = "checkpoints/sac_curiosity"
```

## 5. Danh gia nhieu seed

Script `evaluate_seeds.py` dung de danh gia ket qua on dinh hon qua nhieu seed va xuat CSV.

So sanh SAC + Curiosity, SAC only va Random Agent:

```powershell
.\.venv\Scripts\python.exe evaluate_seeds.py --seeds 0,1,2,3,4 --episodes 50 --max-episode-steps 400 --agent "SAC + Curiosity=checkpoints\sac_curiosity" --agent "SAC only=checkpoints\sac_only"
```

File ket qua:

- `results\evaluation_seeds.csv`: ket qua tung seed
- `results\evaluation_summary.csv`: mean/std theo tung agent

Neu chi muon danh gia nhanh:

```powershell
.\.venv\Scripts\python.exe evaluate_seeds.py --seeds 0 --episodes 5 --max-episode-steps 100 --agent "SAC + Curiosity=checkpoints\sac_curiosity"
```

## 6. Xem TensorBoard

Trong luc train hoac sau khi train, chay:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```

Mo URL TensorBoard hien ra trong terminal, thuong la:

```text
http://localhost:6006
```

Nhung scalar quan trong:

- `reward/train`: reward theo episode
- `success/train`: episode co den dich hay khong
- `steps/train`: so step trong episode
- `loss/critic_1`, `loss/critic_2`, `loss/policy`
- `loss/prediction_loss`: chi co y nghia khi bat curiosity

## 7. Xuat TensorBoard ra CSV

Xuat metric train cua SAC + Curiosity:

```powershell
.\.venv\Scripts\python.exe export_tensorboard_metrics.py --run-filter sac_curiosity --output-csv results\sac_curiosity_training.csv
```

Xuat metric train cua SAC only:

```powershell
.\.venv\Scripts\python.exe export_tensorboard_metrics.py --run-filter sac_only --output-csv results\sac_only_training.csv
```

Script mac dinh xuat 3 tag:

- `reward/train`
- `success/train`
- `steps/train`

Neu trong `runs/` co event file qua lon, script se bo qua file lon hon 100 MB. Co the doi gioi han:

```powershell
.\.venv\Scripts\python.exe export_tensorboard_metrics.py --max-file-mb 500 --output-csv results\all_training.csv
```

## 8. Ve bieu do PNG

Neu muon script tu ve PNG, cai them `matplotlib`:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

Sau do chay:

```powershell
.\.venv\Scripts\python.exe export_tensorboard_metrics.py --run-filter sac_curiosity --output-csv results\sac_curiosity_training.csv --plot-dir results\plots
```

Neu khong cai `matplotlib`, van co the dung file CSV de ve bieu do bang Excel, Google Sheets hoac Python rieng.

## 9. Goi y ket qua nen dua vao bao cao

Nen bao cao it nhat cac bang/bieu do sau:

1. Bang so sanh multi-seed:
   - Random Agent
   - SAC only
   - SAC + Curiosity

2. Metric:
   - `success_rate mean +/- std`
   - `avg_steps mean +/- std`
   - `avg_reward mean +/- std`

3. Bieu do training:
   - episode vs `reward/train`
   - episode vs `success/train`
   - episode vs `steps/train`

4. Nhan xet:
   - Random Agent gan nhu khong den dich trong maze lon.
   - SAC only la baseline hoc tang cuong.
   - SAC + Curiosity nen co success rate cao hon hoac hoc nhanh hon neu curiosity giup kham pha tot hon.

