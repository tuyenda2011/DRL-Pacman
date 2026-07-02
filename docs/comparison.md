# So Sánh Q-Learning, DQN và Double DQN cho Pacman Mini

## 1. Bài Toán

Pacman mini là một môi trường dạng lưới (grid-world). Agent điều khiển Pacman di ăn thức ăn, tránh ma, và tối đa hoá tổng reward.

Trong project này:

- **Trạng thái tabular** (Q-Learning): vị trí Pacman, vị trí ma, số mạng còn lại và food mask (bitmask).
- **Trạng thái vector** (DQN / Double DQN): toạ độ đã chuẩn hoá, khoảng cách tương đối đến thức ăn gần nhất / ma gần nhất, danh sách thức ăn còn lại, và các đặc trưng phụ trợ (hướng nguy hiểm, ô tường xung quanh).
- **Hành động**: 4 hướng — lên, phải, xuống, trái.
- **Reward**: ăn thức ăn được thưởng, thắng được thưởng lớn, chạm ma bị phạt lớn, mỗi bước bị phạt nhỏ, reward shaping theo khoảng cách đến thức ăn và ma.
- **Bản đồ**: `15×15`, `62` thức ăn, `3` ma.
- **Mỗi episode** có `3` mạng; bị ma bắt thì reset vị trí Pacman/ma và giữ thức ăn đã ăn; hết mạng mới tính là `caught`.

### Môi Trường Đặc Biệt

Ma được mô phỏng theo hành vi arcade gốc:

| Tên | Hành vi |
|---|---|
| **Blinky** | Đuổi trực tiếp vị trí Pacman |
| **Pinky** | Chặn đầu — nhắm 4 ô phía trước hướng di chuyển của Pacman |
| **Inky** | Dùng vector qua Blinky và điểm 2 ô trước Pacman để tính mục tiêu |

Ma chuyển đổi giữa chế độ **scatter** (về góc) và **chase** (đuổi) theo lịch cố định, giống game gốc.

---

## 2. Tối Ưu Hiệu Năng — Precomputed BFS Distance Matrix

Để tính khoảng cách mê cung (maze distance) chính xác mà không làm chậm quá trình train, môi trường chạy BFS **một lần duy nhất** từ mỗi ô tự do khi khởi tạo, xây dựng ma trận khoảng cách đầy đủ:

```
Thời gian: O(|ô mở| × (V + E)) — trả một lần khi init
Truy vấn:  O(1) — trong mỗi bước của episode
Bộ nhớ:   O(|ô mở|²) ≈ 130² ≈ 17.000 phần tử cho bản đồ 15×15
```

Nhờ đó, cả khoảng cách đến **ma** và khoảng cách đến **thức ăn gần nhất** đều dùng khoảng cách maze thực tế (không phải Manhattan), đảm bảo reward shaping phản ánh đúng hành vi trong mê cung.

---

## 3. Q-Learning

Q-Learning lưu trực tiếp giá trị `Q(state, action)` trong bảng (Q-Table).

**Công thức cập nhật:**

```
Q(s, a) ← Q(s, a) + α × (r + γ × max_a' Q(s', a') − Q(s, a))
```

**Ưu điểm:**
- Dễ cài đặt, dễ giải thích.
- Hội tụ nhanh khi không gian trạng thái nhỏ.
- Làm baseline rõ ràng để so sánh với DQN.

**Nhược điểm:**
- Không mở rộng tốt khi lưới lớn, nhiều thức ăn, nhiều ma.
- Cần rời rạc hoá trạng thái — không tổng quát hoá giữa các trạng thái lân cận.
- Kích thước Q-Table tăng theo số trạng thái đã thăm.

---

## 4. DQN (Deep Q-Network)

DQN thay Q-Table bằng mạng nơ-ron để ước lượng `Q(s, a; θ)`.

**Các thành phần chính:**
- **Replay Buffer**: lưu trữ kinh nghiệm `(s, a, r, s', done)` và lấy mẫu ngẫu nhiên để phá vỡ tương quan giữa các bước liên tiếp.
- **Target Network**: bản sao của Q-Network được đồng bộ định kỳ, giúp mục tiêu học ổn định hơn.
- **Epsilon-greedy**: cân bằng khám phá (explore) và khai thác (exploit).
- **Gradient clipping**: ngăn gradient bùng nổ.
- **Huber Loss** (`SmoothL1Loss`): ít nhạy cảm với outlier hơn MSE.

**Target của DQN:**

```
y = r + γ × max_a' Q_target(s', a')
```

**Ưu điểm:**
- Làm việc tốt hơn với không gian trạng thái lớn hoặc vector/ảnh.
- Có khả năng tổng quát hoá giữa các trạng thái gần nhau.
- Phù hợp khi vector trạng thái phức tạp hơn trạng thái tabular.

**Nhược điểm:**
- Cần nhiều dữ liệu và điều chỉnh siêu tham số hơn Q-Learning.
- Có thể học không ổn định.
- Dễ đánh giá Q-value quá cao (overestimation bias) do dùng phép `max` trong target.

---

## 5. Double DQN (DDQN)

Double DQN tách việc **chọn hành động** và **đánh giá hành động** ra hai mạng khác nhau:

```
a* = argmax_a Q_online(s', a)       ← online network chọn hành động
y  = r + γ × Q_target(s', a*)      ← target network đánh giá hành động đó
```

**Điểm khác với DQN:**
- **DQN**: target network vừa chọn hành động tốt nhất, vừa đánh giá giá trị của hành động đó → dễ overestimate.
- **Double DQN**: online network chọn, target network đánh giá → giảm overestimation bias.

**Ưu điểm:**
- Giảm overestimation bias một cách có hệ thống.
- Thường ổn định hơn DQN, đặc biệt khi reward nhiễu và môi trường có tính ngẫu nhiên.
- Phù hợp khi ghost có mục tiêu thông minh và môi trường khó hơn random ghost.

**Nhược điểm:**
- Phức tạp hơn DQN một chút.
- Vẫn cần replay buffer, target network và điều chỉnh siêu tham số.

---

## 6. Bảng So Sánh

| Tiêu chí | Q-Learning | DQN | Double DQN |
|---|---|---|---|
| Kiểu hàm Q | Bảng tra cứu | Mạng nơ-ron | Mạng nơ-ron |
| Đầu vào | Trạng thái rời rạc | Vector số thực | Vector số thực |
| Khả năng mở rộng | Thấp | Cao | Cao |
| Ổn định khi train | Cao (với state nhỏ) | Trung bình | Tốt hơn DQN |
| Overestimation bias | Có thể có | Rõ hơn | Giảm đáng kể |
| Yêu cầu tính toán | Thấp | Cao | Cao |
| Độ khó cài đặt | Dễ | Trung bình | Trung bình |
| Vai trò trong đồ án | Baseline tabular | Deep RL baseline | Biến thể cải tiến |

---

## 7. Kỳ Vọng Kết Quả Trên Bản Đồ 15×15

1. **Q-Learning** có thể hội tụ nhanh và ổn định khi không gian trạng thái còn quản lý được.
2. **DQN** có thể cần nhiều episode hơn để vượt Q-Learning, nhưng tổng quát hoá tốt hơn.
3. **Double DQN** thường cho reward trung bình ổn định hơn DQN, nhất là khi ghost dùng AI thông minh (`ghost_chase_probability=1.0`).
4. Khi đánh giá kết quả, nên đọc kết hợp `food_eaten`, `completion_rate` và `win_rate`, không chỉ nhìn vào reward vì reward shaping có thể làm lệch hướng đọc kết quả.

---

## 8. Cách Trình Bày Trong Báo Cáo

Nên chia kết quả theo các mục:

- Đường reward trung bình mỗi 50 hoặc 100 episode (moving average).
- Tỉ lệ thắng (`win_rate`).
- Số bước trung bình mỗi episode.
- Tỉ lệ hoàn thành (`completion_rate`) — phần trăm thức ăn đã ăn.
- Nhận xét về độ ổn định của quá trình huấn luyện.

**Kết luận ngắn gọn:**

- Q-Learning là baseline tốt cho Pacman mini với bản đồ nhỏ.
- DQN mở rộng tốt hơn khi không gian trạng thái phức tạp.
- Double DQN là cải tiến của DQN, thường ổn định hơn và giảm việc đánh giá Q-value quá cao.
