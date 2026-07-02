# DRL-Pacman

Project so sanh 3 cach tiep can Reinforcement Learning cho bai toan Pacman mini:

- `Q-learning`: baseline tabular, de hieu, phu hop khi state space con nho.
- `DQN`: dung neural network de uoc luong Q-value.
- `Double DQN`: bien the DQN giup giam overestimation bias.

Moi truong mac dinh la map Pacman `15x15`, co `62` food, `3` ghost, `3` mang, wall, reward, episode timeout va GUI de xem model da train choi lai. Project tap trung vao map 15x15 de de so sanh 3 thuat toan tren cung mot bai toan; maze duoc thiet ke gan map Pac-Man goc hon voi layout doi xung, vung ghost o trung tam, Pacman xuat phat phia duoi va pellet phu day hanh lang. Ghost co ten chinh thuc trong code la `Blinky`, `Pinky`, `Inky`: Blinky duoi truc tiep, Pinky chan dau, Inky dung vector target, co scatter/chase mode, chon duong bang BFS theo maze va mac dinh khong di random. Khi bi ghost bat, Pacman mat 1 mang, reset Pacman/ghost ve vi tri xuat phat va giu lai food da an; het 3 mang moi ket thuc episode voi `caught`. GUI duoc doi theo phong cach `graphicsDisplay.py`: nen den, wall xanh Berkeley, pellet/capsule trang, Pacman vang va ghost shape/mau Berkeley.

## Cau Truc

```text
DRL-Pacman/
|-- configs/
|   |-- q_learning/
|   |   |-- q_learning_lr_0001.yaml
|   |   |-- q_learning_lr_001.yaml
|   |   `-- q_learning_lr_01.yaml
|   |-- dqn/
|   |   |-- dqn_lr_0001.yaml
|   |   |-- dqn_lr_001.yaml
|   |   `-- dqn_lr_01.yaml
|   `-- double_dqn/
|       |-- double_dqn_lr_0001.yaml
|       |-- double_dqn_lr_001.yaml
|       `-- double_dqn_lr_01.yaml
|-- docs/
|   `-- comparison.md
|-- experiments/
|   |-- metrics/
|   |-- history/
|   `-- plots/
|-- models/
|   |-- final/
|   |   |-- q_learning/
|   |   |-- dqn/
|   |   `-- double_dqn/
|   `-- checkpoints/
|       |-- q_learning/
|       |-- dqn/
|       `-- double_dqn/
|-- src/
|   |-- algorithms/
|   |-- pacman_env/
|   |   `-- grid_world.py
|   `-- training/
|       |-- checkpointing.py
|       |-- compare_runs.py
|       |-- logging_utils.py
|       |-- train_q_learning.py
|       |-- train_dqn.py
|       |-- train_double_dqn.py
|       `-- watch_model.py
`-- tests/
```

`configs/<algorithm>/<algorithm>_lr_*.yaml` la config theo learning rate cho tung script train. Mac dinh cac script doc file `*_lr_001.yaml`, vi day la learning rate chinh de so sanh 3 thuat toan; command-line flags dung de override khi can tuning nhanh.

Cac config mac dinh duoc can bang de de so sanh: `episodes`, `learning_rate`, `discount_factor`, `epsilon_start`, `epsilon_min`, `epsilon_decay`, map, ghost, seed, checkpoint va evaluation interval duoc giu giong nhau giua 3 thuat toan. Cac tham so chi co o DQN/Double DQN nhu `batch_size`, `replay_capacity`, `learning_starts`, `target_update_interval` va `hidden_size` la phan rieng cua neural network. Khi can toi uu ket qua thuc te, co the tune rieng tung thuat toan bang CLI flags.

## Cai Dat

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

DQN va Double DQN can `torch`. Neu chi train Q-learning thi khong can GPU.

## Train

Q-learning:

```bash
python -m src.training.train_q_learning
```

DQN:

```bash
python -m src.training.train_dqn
```

Double DQN:

```bash
python -m src.training.train_double_dqn
```

Moi script tu load config mac dinh:

```text
train_q_learning.py  -> configs/q_learning/q_learning_lr_001.yaml
train_dqn.py         -> configs/dqn/dqn_lr_001.yaml
train_double_dqn.py  -> configs/double_dqn/double_dqn_lr_001.yaml
```

Muon override mot vai tham so thi truyen them flag:

```bash
python -m src.training.train_dqn --episodes 3000 --ghost-count 3
```

Muon chi ro config khac:

```bash
python -m src.training.train_dqn --config configs/dqn/dqn_lr_001.yaml --episodes 1000
```

So sanh learning rate thi dung config rieng trong tung folder thuat toan:

```bash
python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_0001.yaml
python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_001.yaml
python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_01.yaml
```

Hoac chay 3 thuat toan theo tung learning rate:

```powershell
.\scripts\run_lr_0001.ps1
.\scripts\run_lr_001.ps1
.\scripts\run_lr_01.ps1
```

Cac file tao ra:

- Metrics tung episode: `experiments/metrics/*_metrics.csv`
- Metrics danh gia policy voi `epsilon=0`: `experiments/metrics/*_eval_metrics.csv`
- Model cuoi cung: `models/final/<algorithm>/<model_file>`
- Checkpoint de resume: `models/checkpoints/<algorithm>/<run_name>/*_checkpoint_epXXXX.pkl`
- Lich su moi lan train: `experiments/history/training_runs.jsonl`

CSV metrics duoc ghi streaming tung episode, nen train lau hoac bi dung giua chung van giu du lieu da ghi.

## Checkpoint Va Resume

Checkpoint da duoc tach rieng trong:

```text
src/training/checkpointing.py
```

Trainer chi goi helper save/load checkpoint, con agent chi lo logic thuat toan va save/load model.

Mac dinh checkpoint duoc luu moi `1000` episode:

```text
models/checkpoints/q_learning/q_learning_lr_001/q_learning_lr_001_checkpoint_ep1000.pkl
models/checkpoints/dqn/dqn_lr_001/dqn_lr_001_checkpoint_ep1000.pkl
models/checkpoints/double_dqn/double_dqn_lr_001/double_dqn_lr_001_checkpoint_ep1000.pkl
```

`checkpoint_path` trong config la ten goc, vi du `models/checkpoints/dqn/dqn_lr_001/dqn_lr_001_checkpoint.pkl`. Khi save, code tu them episode vao ten file: `dqn_lr_001_checkpoint_ep1000.pkl`, `dqn_lr_001_checkpoint_ep2000.pkl`, ... Nen checkpoint cu khong bi overwrite va moi cau hinh learning rate co folder rieng.

Checkpoint luu trang thai de train tiep:

- Q-learning: `episode`, `elapsed_sec`, `win_count`, `epsilon`, random state, Q-table.
- DQN/Double DQN: cac truong tren, them `global_step`, online network, target network, optimizer, replay buffer.

Train tiep tu checkpoint:

```bash
python -m src.training.train_q_learning --resume
python -m src.training.train_dqn --resume
python -m src.training.train_double_dqn --resume
```

`--episodes` la tong episode muc tieu, khong phai so episode cong them. Vi du checkpoint dang o episode `1200`, chay `--episodes 3000 --resume` se train tiep tu `1201` den `3000`.

Neu `--checkpoint-path` tro toi ten goc nhu `models/checkpoints/dqn/dqn_lr_001/dqn_lr_001_checkpoint.pkl`, resume se tu tim checkpoint episode moi nhat cung prefix, vi du `dqn_lr_001_checkpoint_ep1000.pkl`.

Dung file checkpoint rieng khi train nhieu cau hinh:

```bash
python -m src.training.train_dqn --ghost-count 3 --checkpoint-path models/checkpoints/dqn/dqn_lr_001_ghost3/dqn_lr_001_ghost3_checkpoint.pkl
python -m src.training.train_dqn --ghost-count 3 --checkpoint-path models/checkpoints/dqn/dqn_lr_001_ghost3/dqn_lr_001_ghost3_checkpoint.pkl --resume
```

## Dung Ngang Khi Dang Train

Neu dang train va muon dung lai, hay bam `Ctrl+C` trong terminal.

Code se:

- ghi episode hien tai vao CSV voi `event=interrupted`
- khong tao checkpoint le giua moc
- resume se quay ve checkpoint theo moc gan nhat, vi du `ep1000`, `ep2000`, ...
- ghi record vao `experiments/history/training_runs.jsonl` voi `status=interrupted`
- khong save model final trong `models/final/`
- thoat gon de lan sau resume bang `--resume`

Luu y: checkpoint chi duoc tao theo `checkpoint_interval`. Neu dung truoc moc checkpoint dau tien thi se chua co checkpoint de resume.

## Training History

Moi lan train xong hoac bi interrupt se append 1 dong JSON vao:

```text
experiments/history/training_runs.jsonl
```

Moi record gom:

- algorithm
- status: `completed`, `interrupted`, hoac `skipped`
- resume true/false
- episode range
- config da chay
- duong dan metrics/model/checkpoint
- final metrics cua episode cuoi

Doi file history neu can:

```bash
python -m src.training.train_q_learning --history-output experiments/history/my_runs.jsonl
```

## DQN Warmup

DQN va Double DQN co replay buffer. Tham so quan trong:

```text
--learning-starts 10000
```

Agent se thu thap it nhat `10000` transition truoc khi update neural network. Dieu nay giup buffer bot ngheo du lieu tren map `15x15` co nhieu food va 3 ghost.

Mac dinh DQN/Double DQN dung cau hinh gon de de hoc va chay nhanh:

```text
batch_size: 64
target_update_interval: 1500
hidden_size: 128
```

Neu muon tuning sau khi da nam thuat toan, override truc tiep bang flag de tranh sinh qua nhieu config.

Vi du:

```bash
python -m src.training.train_dqn --episodes 10000 --learning-starts 10000
```

## Evaluation Khong Exploration

Moi trainer co the chay danh gia dinh ky voi `epsilon=0` de do policy that su, tach rieng khoi metrics training van con exploration.

Mac dinh trong `configs/*.yaml`, evaluation chay moi `1000` episode voi `20` episode eval:

```text
eval_interval: 1000
eval_episodes: 20
eval_output: experiments/metrics/<algorithm>_eval_metrics.csv
eval_seed: 10042
```

Override nhanh:

```bash
python -m src.training.train_dqn --eval-interval 500 --eval-episodes 50
python -m src.training.train_double_dqn --eval-interval 500 --eval-episodes 50
python -m src.training.train_q_learning --eval-interval 500 --eval-episodes 50
```

CSV eval gom cac cot tong hop theo moc episode: `avg_reward`, `win_rate`, `avg_food_eaten`, `avg_completion_rate`, `wins`, `caught`, `timeout`.

Danh gia mac dinh chi dung map `medium` 15x15 voi `3` ghost, `3` mang va `62` food. Moi truong khong cho `4` ghost vi map 15x15 se qua chat.

Goi y tuning nhanh neu can:

```bash
python -m src.training.train_dqn --learning-rate 0.0003 --target-update-interval 1000 --hidden-size 256
python -m src.training.train_double_dqn --learning-rate 0.0003 --target-update-interval 1000 --hidden-size 256
python -m src.training.train_q_learning --learning-rate 0.05 --epsilon-decay 0.9999
```

Muon so sanh nhieu seed thi them `--seed 123` hoac `--seed 2026` va doi `--output`/`--eval-output` de khong ghi de file cu.

## Xem Model Choi

GUI nam o:

```text
src/training/watch_model.py
```

Giao dien da duoc doi sang phong cach Pac-Man classic theo `graphicsDisplay.py`: nen den, wall xanh Berkeley, HUD `1UP / HIGH SCORE`, pellet/capsule trang, ghost than tron chan luon song, cherry bonus tren map va cum cherry phia duoi HUD.

Chi xem map ban dau tren GUI, khong can model va khong train:

```bash
python -m src.training.watch_model --map-only
```

Demo ghost chay tren map, Pacman bi an tren man hinh va khong can model:

```bash
python -m src.training.watch_model --ghost-demo --delay-ms 500
```

Xem Q-learning:

```bash
python -m src.training.watch_model --algorithm q_learning --ghost-count 3
```

Xem DQN:

```bash
python -m src.training.watch_model --algorithm dqn --ghost-count 3
```

Xem Double DQN:

```bash
python -m src.training.watch_model --algorithm double_dqn --ghost-count 3
```

Chi ro file model:

```bash
python -m src.training.watch_model --algorithm q_learning --model-path models/final/q_learning/q_learning.pkl
```

Tuy chinh toc do/kich thuoc:

```bash
python -m src.training.watch_model --algorithm q_learning --episodes 5 --delay-ms 500 --cell-size 30
```

Mac dinh GUI dung `--delay-ms 500` de de quan sat model choi. Muon xem nhanh hon co the giam xuong `250` hoac `150`.

Luu y: state vector hien co them so mang con lai, nen model/checkpoint DQN hoac Double DQN train truoc thay doi nay can train lai de khop input size.

Model nen duoc xem voi cung cau hinh luc train, vi du `--ghost-count`, `--max-lives` va `--ghost-chase-probability`.
`watch_model` se tu doc `configs/<algorithm>/<algorithm>_lr_001.yaml`, nen command mac dinh se khop `ghost_count`, `max_lives` va `ghost_chase_probability` cua luc train chinh. Mac dinh ca 3 thuat toan dung map 15x15, `3` ghost, `3` mang va `ghost_chase_probability=1.0` de ghost di theo target thong minh nhu game goc.

## Log Terminal

Them `--log-interval 1` de in tung episode:

```bash
python -m src.training.train_q_learning --episodes 50 --log-interval 1
python -m src.training.train_dqn --episodes 50 --log-interval 1
python -m src.training.train_double_dqn --episodes 50 --log-interval 1
```

Them `--render-interval` de in map text len terminal:

```bash
python -m src.training.train_dqn --episodes 50 --log-interval 1 --render-interval 10
```

Y nghia metric:

- `reward`: tong diem cua episode.
- `steps`: so buoc da di.
- `epsilon`: ti le exploration.
- `loss`: loss neural network, chi co voi DQN/Double DQN.
- `event`: trang thai ket thuc episode la `win`, `caught`, `timeout`, hoac `interrupted`; rieng `life_lost` la event trung gian khi Pacman bi bat nhung van con mang.
- `win_rate`: ti le thang tinh den episode hien tai.
- `food_eaten`: so food an duoc.
- `completion_rate`: ti le food da an.
- `elapsed_sec`: thoi gian train da chay.
- `q_states`: so state trong Q-table.
- `buffer_size`: kich thuoc replay buffer.
- `global_step`: tong so environment step cua DQN/Double DQN.

Reward trong moi truong:

- moi step: `-0.1`
- song sot qua step: `+0.02`
- dam vao wall: `-1`
- tien gan food gan nhat: `+0.03`
- di xa food gan nhat: `-0.03`
- tang khoang cach toi ghost gan nhat: `+0.05`
- giam khoang cach toi ghost gan nhat khi ghost dang gan: `-0.10`
- an food: `+10`
- bi bat mat 1 mang: `-50`
- timeout: `-10`
- an het food: `+50`

## Ve Bieu Do So Sanh

Sau khi co metrics cua cac thuat toan:

```bash
python -m src.training.compare_runs
```

Output mac dinh gom reward, completion rate va win-rate:

```text
experiments/plots/training_comparison.png
```

Co the ve tu eval CSV:

```bash
python -m src.training.compare_runs --output experiments/plots/eval_comparison.png experiments/metrics/q_learning_eval_metrics.csv experiments/metrics/dqn_eval_metrics.csv experiments/metrics/double_dqn_eval_metrics.csv
```

## Test

```bash
python -m pytest
```

Test hien co cover:

- environment behavior
- Pac-Man style ghost target logic va scatter/chase mode
- CSV/history logging
- config YAML va CLI override
- evaluation voi epsilon=0 va restore epsilon sau khi danh gia
- Q-learning checkpoint round-trip
- DQN target calculation, mini-batch update smoke test
- Double DQN target calculation: online network chon action, target network danh gia
- DQN checkpoint round-trip: online network, target network, optimizer, replay buffer, epsilon

## Goi Y Danh Gia

Voi map medium `15x15`, `62` food va `3` ghost, ca 3 thuat toan duoc so sanh tren cung mot moi truong co nhieu pellet va ghost thong minh hon random. Khi danh gia, dung them `food_eaten` va `completion_rate`, khong chi nhin `win_rate`.

DQN va Double DQN co kha nang tong quat tot hon, nhung can replay buffer, warmup va nhieu episode hon. Double DQN thuong on dinh hon DQN do tach buoc chon action va danh gia target.

Neu bo qua viec train dai de lay ket qua cuoi, repo van co the duoc danh gia theo phan implementation:

- 3 thuat toan co vai tro ro: Q-learning baseline, DQN, Double DQN.
- DQN dung replay buffer, target network, Huber loss va gradient clipping.
- Double DQN co test rieng cho cong thuc target khac DQN.
- Metrics, eval, checkpoint, resume va history log duoc tach thanh module rieng.
- Config YAML giup tai lap thi nghiem va CLI flags giup tuning nhanh.
- GUI chi dung de xem model da train, khong bi tron vao logic training.
- Test khong phu thuoc train lau, tap trung vao logic thuat toan va ha tang thuc nghiem.

Phan phan tich chi tiet nam o:

```text
docs/comparison.md
```
