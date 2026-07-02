# DRL-Pacman

Project so sánh 3 cách tiếp cận Reinforcement Learning cho bài toán Pacman mini:
- `Q-learning`: Baseline tabular, dễ hiểu, phù hợp khi state space nhỏ.
- `DQN`: Dùng neural network để ước lượng Q-value, khắc phục hạn chế về bộ nhớ của Q-Table.
- `Double DQN`: Biến thể DQN giúp giảm thiểu hiện tượng đánh giá quá mức (overestimation bias).

Môi trường mặc định là map Pacman `15x15`, có `62` food, `3` ghost, `3` mạng. Ghost thông minh được mô phỏng theo game gốc: `Blinky` (đuổi trực tiếp), `Pinky` (chặn đầu), `Inky` (chọn điểm chặn qua vector). Môi trường đã được tinh chỉnh tối ưu hiệu năng để train với tốc độ cực cao (sử dụng ma trận BFS tính sẵn).

---

## 🛠 Cấu Trúc Project

```text
DRL-Pacman/
|-- configs/               <-- File cấu hình siêu tham số (Hyperparameters)
|   |-- q_learning/
|   |-- dqn/
|   `-- double_dqn/
|-- experiments/           <-- Dữ liệu train và biểu đồ
|   |-- metrics/           <-- Các file CSV ghi lại Reward, Win rate...
|   |-- history/           <-- Lịch sử các lần chạy
|   `-- plots/             <-- Biểu đồ so sánh
|-- models/                <-- Trọng số / Q-Table đã train
|   |-- final/             <-- Model cuối cùng
|   `-- checkpoints/       <-- Model lưu định kỳ
|-- src/
|   |-- algorithms/        <-- Core logic 3 thuật toán
|   |-- pacman_env/        <-- Môi trường Mini Pacman
|   `-- training/          <-- Script train, đánh giá, log, GUI
|-- run_all_experiments.py <-- Script chạy tự động tất cả
`-- tests/
```

---

## 🚀 Cài Đặt

Môi trường yêu cầu `Python 3.10+` và cài đặt qua `requirements.txt`:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
*(Lưu ý: Nếu bạn sử dụng conda, hãy dùng lệnh `conda activate DRL`)*

---

## ⚙️ Hướng Dẫn Train

Hệ thống của project hỗ trợ tính năng **Smart Checkpoint (Phân luồng thông minh)**. Bạn không cần phải lo về việc đặt tên file output hay lo sợ bị ghi đè dữ liệu cũ. Toàn bộ đường dẫn lưu Metrics, Models, History đều sẽ được trích xuất tự động dựa trên **Tên File YAML** cấu hình.

### Cách 1: Chạy Tự Động (Khuyên Dùng)

Tôi đã thiết lập sẵn một script chạy 1-click. Lệnh này sẽ tự động chạy lần lượt cả 3 thuật toán (Q-Learning, DQN, Double DQN) theo đúng thứ tự, và cuối cùng tự động vẽ biểu đồ so sánh:

```bash
python run_all_experiments.py
```

### Cách 2: Chạy Thủ Công Từng Thuật Toán

Bạn có thể tự gọi các script train và chỉ định file cấu hình `.yaml`:

```bash
python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_01.yaml
python -m src.training.train_dqn --config configs/dqn/dqn_lr_001.yaml
python -m src.training.train_double_dqn --config configs/double_dqn/double_dqn_lr_001.yaml
```

Ví dụ, khi bạn chỉ định file `dqn_lr_001.yaml`, hệ thống sẽ hiểu tên lượt chạy (run_name) là `dqn_lr_001` và tự động lưu vào các thư mục:
- **CSV Output:** `experiments/metrics/dqn/dqn_lr_001_metrics.csv`
- **History:** `experiments/history/dqn/dqn_lr_001_history.jsonl`
- **Model Final:** `models/final/dqn/dqn_lr_001.pt`
- **Checkpoints:** `models/checkpoints/dqn/dqn_lr_001/dqn_lr_001_checkpoint_epXXXX.pkl`

*(Nếu bạn muốn đổi tham số như Learning Rate, Epsilon, Layout... chỉ cần copy tạo file YAML mới và chạy. File output sẽ tự động rẽ sang một thư mục mới tinh).*

---

## 🔄 Checkpoint & Resume (Tiếp Tục Train)

Mặc định hệ thống tự lưu checkpoint mỗi `500` - `1000` episode. Nếu bạn bấm `Ctrl+C` giữa chừng, hệ thống sẽ dừng an toàn. 
Để train tiếp từ chỗ bị đứt đoạn, chỉ cần thêm cờ `--resume`:

```bash
python -m src.training.train_dqn --config configs/dqn/dqn_lr_001.yaml --resume
```

---

## 📊 Vẽ Biểu Đồ So Sánh

Khi bạn muốn so sánh kết quả (Reward, Tỷ lệ hoàn thành, Win Rate) của các lượt train với nhau, chỉ cần gọi script `compare_runs.py`:

```bash
python -m src.training.compare_runs
```

Script đã được nâng cấp để **Tự động quét toàn bộ thư mục `experiments/metrics/`** và vẽ mọi file CSV nó tìm thấy thành biểu đồ so sánh (được xuất ra tại `experiments/plots/training_comparison.png`).

*(Lưu ý: Nếu bạn chỉ muốn so sánh một vài file cụ thể, bạn có thể truyền đường dẫn của chúng đằng sau lệnh trên).*

---

## 🎮 Xem Model Chơi (GUI)

Bạn có thể xem trực tiếp Model của mình đã học được gì thông qua giao diện PyGame:

```bash
# Xem Q-Learning
python -m src.training.watch_model --algorithm q_learning

# Xem DQN
python -m src.training.watch_model --algorithm dqn

# Xem Double DQN
python -m src.training.watch_model --algorithm double_dqn
```

Script sẽ tự động tìm kiếm file model final hoặc checkpoint mới nhất của thuật toán tương ứng để biểu diễn. Giao diện được thiết kế theo đúng phong cách Pac-Man cổ điển (nền đen, tường xanh Berkeley).

---

## 📝 Phân Tích & Đánh Giá

Chi tiết về độ phức tạp bộ nhớ, hiện tượng Overestimation Bias, và quá trình tối ưu hiệu năng của dự án đã được ghi chép và phân tích kỹ lưỡng ở file:
`docs/comparison.md`
