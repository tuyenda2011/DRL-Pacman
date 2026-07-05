# DRL-Pacman

DRL-Pacman là project thực nghiệm Reinforcement Learning trên môi trường Mini Pacman. Mục tiêu là huấn luyện, đánh giá và so sánh 3 thuật toán:

- `Q-learning`: baseline dạng Q-table.
- `DQN`: deep Q-network dùng neural network để xấp xỉ Q-value.
- `Double DQN`: biến thể DQN giảm overestimation bias khi chọn action.

Project đã có đầy đủ pipeline: cấu hình train bằng YAML, checkpoint/resume, final model, metric CSV, biểu đồ so sánh và GUI xem agent chơi.

## Kết Quả Hiện Tại

Toàn bộ 9 run đã train tới `20000` episodes:

```text
3 thuật toán x 3 learning rate = 9 runs
learning rates: 0.0001, 0.0005, 0.001
```

Bảng dưới đây lấy từ file `*_eval_metrics.csv` ở episode `20000`, với `20` evaluation episodes mỗi run.

| Algorithm | LR | Avg reward | Avg completion | Win rate | Avg steps | Best completion |
|---|---:|---:|---:|---:|---:|---:|
| Double DQN | 0.0001 | 1247.154 | 99.76% | 95% | 148.30 | 100% |
| Double DQN | 0.0005 | 1312.231 | 99.27% | 85% | 153.60 | 100% |
| Double DQN | 0.001 | 1553.422 | 99.27% | 65% | 211.65 | 100% |
| DQN | 0.0001 | 1125.694 | 98.55% | 75% | 157.85 | 100% |
| DQN | 0.0005 | 1254.443 | 99.84% | 95% | 138.90 | 100% |
| DQN | 0.001 | 1148.294 | 98.71% | 75% | 161.85 | 100% |
| Q-learning | 0.0001 | 61.960 | 26.29% | 0% | 99.25 | 51.61% |
| Q-learning | 0.0005 | 134.079 | 42.66% | 0% | 124.05 | 51.61% |
| Q-learning | 0.001 | 165.472 | 46.29% | 0% | 134.40 | 54.84% |

Tóm tắt nhanh:

- `DQN` và `Double DQN` học tốt hơn rõ rệt so với `Q-learning` trong môi trường hiện tại.
- `DQN lr=0.0005` và `Double DQN lr=0.0001` đạt win rate `95%`.
- `Double DQN lr=0.001` có avg reward cao nhất, nhưng win rate thấp hơn hai run tốt nhất.
- `Q-learning` không clear map trong eval cuối, nhưng completion vẫn tăng khi learning rate cao hơn.

## Biểu Đồ Kết Quả

### Reward

| LR 0.0001 | LR 0.0005 | LR 0.001 |
|---|---|---|
| ![Reward 0.0001](experiments/plots/reward_0001.png) | ![Reward 0.0005](experiments/plots/reward_0005.png) | ![Reward 0.001](experiments/plots/reward_001.png) |

### Completion Rate

| LR 0.0001 | LR 0.0005 | LR 0.001 |
|---|---|---|
| ![Completion 0.0001](experiments/plots/completion_0001.png) | ![Completion 0.0005](experiments/plots/completion_0005.png) | ![Completion 0.001](experiments/plots/completion_001.png) |

### Win Rate

| LR 0.0001 | LR 0.0005 | LR 0.001 |
|---|---|---|
| ![Win rate 0.0001](experiments/plots/win_rate_0001.png) | ![Win rate 0.0005](experiments/plots/win_rate_0005.png) | ![Win rate 0.001](experiments/plots/win_rate_001.png) |

### Steps

| LR 0.0001 | LR 0.0005 | LR 0.001 |
|---|---|---|
| ![Steps 0.0001](experiments/plots/steps_0001.png) | ![Steps 0.0005](experiments/plots/steps_0005.png) | ![Steps 0.001](experiments/plots/steps_001.png) |

### Loss

`loss` chỉ áp dụng cho `DQN` và `Double DQN`.

| LR 0.0001 | LR 0.0005 | LR 0.001 |
|---|---|---|
| ![Loss 0.0001](experiments/plots/loss_0001.png) | ![Loss 0.0005](experiments/plots/loss_0005.png) | ![Loss 0.001](experiments/plots/loss_001.png) |

## Biểu Đồ Đã Sinh

Biểu đồ nằm trong:

```text
experiments/plots/
```

Mặc định `compare_runs.py` sinh `15` ảnh:

```text
completion_0001.png
completion_0005.png
completion_001.png
loss_0001.png
loss_0005.png
loss_001.png
reward_0001.png
reward_0005.png
reward_001.png
steps_0001.png
steps_0005.png
steps_001.png
win_rate_0001.png
win_rate_0005.png
win_rate_001.png
```

Ý nghĩa:

- `reward`: hiệu quả tổng thể của policy.
- `completion`: tỉ lệ food/map agent hoàn thành được.
- `win_rate`: tỉ lệ clear map.
- `steps`: agent sống/chơi được bao lâu trong mỗi episode.
- `loss`: chỉ so sánh `DQN` và `Double DQN`, vì `Q-learning` không có neural network loss.

## Cấu Trúc Project

```text
DRL-Pacman/
|-- configs/
|   |-- q_learning/
|   |-- dqn/
|   `-- double_dqn/
|-- experiments/
|   |-- metrics/
|   |-- history/
|   `-- plots/
|-- models/
|   |-- final/
|   `-- checkpoints/
|-- scripts/
|-- src/
|   |-- algorithms/
|   |-- pacman_env/
|   `-- training/
|-- tests/
|-- requirements.txt
`-- run_all_experiments.py
```

## Cài Đặt

Repo hiện được chạy bằng env conda `DRL`.

```powershell
conda activate DRL
pip install -r requirements.txt
```

Hoặc dùng venv:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Dependency chính:

```text
numpy
torch
matplotlib
PyYAML
pytest
```

GUI xem model dùng `tkinter`, thường có sẵn trong Python trên Windows.

## Cấu Hình Train

Mỗi thuật toán có 3 learning rate:

```text
0.0001 -> *_lr_0001.yaml
0.0005 -> *_lr_0005.yaml
0.001  -> *_lr_001.yaml
```

Tổng cộng 9 config:

```text
configs/q_learning/q_learning_lr_0001.yaml
configs/q_learning/q_learning_lr_0005.yaml
configs/q_learning/q_learning_lr_001.yaml
configs/dqn/dqn_lr_0001.yaml
configs/dqn/dqn_lr_0005.yaml
configs/dqn/dqn_lr_001.yaml
configs/double_dqn/double_dqn_lr_0001.yaml
configs/double_dqn/double_dqn_lr_0005.yaml
configs/double_dqn/double_dqn_lr_001.yaml
```

## Chạy Train

Chạy bộ mặc định, tức 3 thuật toán với `lr=0.001`:

```powershell
python run_all_experiments.py
```

Chạy toàn bộ 9 config:

```powershell
python run_all_experiments.py --all-lr
```

Xem trạng thái mà không train:

```powershell
python run_all_experiments.py --all-lr --status
```

Chạy thông minh: skip run đã xong, resume run đang dở, train mới nếu chưa có artifact:

```powershell
python run_all_experiments.py --all-lr --auto-resume
```

Nếu checkpoint cũ không tương thích:

```powershell
python run_all_experiments.py --all-lr --auto-resume --restart-incompatible
```

Chạy riêng một learning rate:

```powershell
.\scripts\run_lr_0001.ps1
.\scripts\run_lr_0005.ps1
.\scripts\run_lr_001.ps1
```

Chạy thủ công một thuật toán:

```powershell
python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_0005.yaml
python -m src.training.train_dqn --config configs/dqn/dqn_lr_0005.yaml
python -m src.training.train_double_dqn --config configs/double_dqn/double_dqn_lr_0005.yaml
```

Resume thủ công:

```powershell
python -m src.training.train_dqn --config configs/dqn/dqn_lr_0005.yaml --resume
```

## Output Sau Train

Ví dụ với:

```text
configs/dqn/dqn_lr_0005.yaml
```

Output mặc định:

```text
experiments/metrics/dqn/dqn_lr_0005_metrics.csv
experiments/metrics/dqn/dqn_lr_0005_eval_metrics.csv
experiments/history/dqn/dqn_lr_0005_history.jsonl
models/final/dqn/dqn_lr_0005.pt
models/checkpoints/dqn/dqn_lr_0005/dqn_lr_0005_checkpoint_epXXXX.pkl
```

Final model hiện đã có đủ cho cả 9 run trong:

```text
models/final/
```

## Vẽ Lại Biểu Đồ

Tạo lại toàn bộ biểu đồ mặc định:

```powershell
python -m src.training.compare_runs
```

Mặc định script đọc train metrics và tạo ảnh theo format:

```text
<metric>_<lr>.png
```

Ví dụ:

```text
reward_0005.png
completion_0005.png
loss_0005.png
```

Vẽ eval metrics:

```powershell
python -m src.training.compare_runs --source eval
```

Vẽ cả train và eval:

```powershell
python -m src.training.compare_runs --source all
```

Gom tất cả vào một ảnh tổng:

```powershell
python -m src.training.compare_runs --combined
```

Chỉ vẽ một vài metric:

```powershell
python -m src.training.compare_runs --metrics reward completion
```

## Xem Model Chơi

Mở launcher chọn final model:

```powershell
python -m src.training.watch_finals
```

Liệt kê model final:

```powershell
python -m src.training.watch_finals --list
```

Xem một thuật toán:

```powershell
python -m src.training.watch_finals --algorithm dqn
```

Chạy lần lượt toàn bộ final model:

```powershell
python -m src.training.watch_finals --all
```

Chỉnh tốc độ GUI:

```powershell
python -m src.training.watch_finals --algorithm double_dqn --all --delay-ms 150
```

Xem bằng config cụ thể:

```powershell
python -m src.training.watch_model --config configs/dqn/dqn_lr_0005.yaml
```

Ưu tiên checkpoint mới nhất thay vì final model:

```powershell
python -m src.training.watch_model --config configs/dqn/dqn_lr_0005.yaml --prefer-checkpoint
```

## Kiểm Tra

Chạy test:

```powershell
pytest
```

Kiểm tra cú pháp script chính:

```powershell
python -m py_compile run_all_experiments.py src/training/watch_model.py src/training/watch_finals.py src/training/compare_runs.py
```

## Ghi Chú

- `loss` chỉ có ý nghĩa với `DQN` và `Double DQN`.
- `win_rate` có thể phẳng nếu agent chưa clear map ổn định; khi đó `reward` và `completion` thường hữu ích hơn để phân tích.
- `steps` giúp phân biệt agent sống lâu thật sự với agent chỉ ăn nhanh rồi chết.
- Phân tích chi tiết hơn nằm ở `docs/comparison.md`.
