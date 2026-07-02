# So sanh Q-learning, DQN va Double DQN cho Pacman mini

## 1. Bai toan

Pacman mini la mot moi truong dang luoi. Agent dieu khien Pacman di an thuc an, tranh ma, va toi da hoa tong reward.

Trong scaffold nay:

- Trang thai tabular gom vi tri Pacman, vi tri ma, so mang con lai va food mask.
- Trang thai vector cho DQN/Double DQN gom vi tri da chuan hoa, so mang con lai va danh sach food con lai.
- Hanh dong gom 4 huong: len, phai, xuong, trai.
- Reward co ban: an food duoc thuong, thang duoc thuong lon, cham ma bi phat lon, moi buoc bi phat nhe.
- Map danh gia mac dinh la `15x15`, `62` food, `3` ghost.
- Moi episode co `3` mang; bi ghost bat thi reset Pacman/ghost va giu food da an, het mang moi tinh la `caught`.
- Config chinh cua moi thuat toan la file `*_lr_001.yaml`, vi du `configs/dqn/dqn_lr_001.yaml`.
- Config so sanh learning rate gom `*_lr_0001.yaml`, `*_lr_001.yaml`, `*_lr_01.yaml`.
- Maze gan map Pac-Man goc hon: doi xung, nhieu pellet, ghost o trung tam va Pacman xuat phat phia duoi.
- Ghost co ten chinh thuc trong code: `Blinky`, `Pinky`, `Inky`.
- Ghost dung hanh vi gan Pac-Man goc: chase/scatter mode, Blinky duoi truc tiep, Pinky chan dau, Inky dung vector target va chon duong theo BFS trong maze.
- Khong dung 4 ghost vi map 15x15 se qua chat va kho danh gia tien do an food.

## 2. Q-learning

Q-learning luu truc tiep gia tri `Q(state, action)` trong bang.

Cong thuc cap nhat:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

Uu diem:

- De cai dat, de giai thich.
- Tot cho Pacman mini neu so trang thai nho.
- Lam baseline ro rang de so voi DQN.

Nhuoc diem:

- Khong mo rong tot khi grid lon, nhieu food, nhieu ma.
- Can roi rac hoa state.
- Khong tong quat hoa giua cac state gan nhau.

## 3. DQN

DQN thay bang Q-table bang neural network:

```text
Q(s, a; theta)
```

DQN thuong dung:

- Replay buffer de lay mau kinh nghiem ngau nhien.
- Target network de tinh target on dinh hon.
- Epsilon-greedy de can bang explore va exploit.

Target cua DQN:

```text
y = r + gamma * max_a' Q_target(s', a')
```

Uu diem:

- Lam viec tot hon voi state lon hoac state vector/anh.
- Co kha nang tong quat hoa.
- Phu hop khi state vector phuc tap hon tabular state va can tong quat hoa giua cac vi tri gan nhau.

Nhuoc diem:

- Can nhieu du lieu va tuning hon Q-learning.
- Co the hoc khong on dinh.
- Hay danh gia Q-value qua cao vi cung dung phep max trong target.

## 4. Double DQN (DDQN)

Double DQN tach viec chon action va danh gia action:

```text
a* = argmax_a Q_online(s', a)
y = r + gamma * Q_target(s', a*)
```

Diem khac voi DQN:

- DQN: target network vua gop phan chon action tot nhat, vua danh gia gia tri cua action do.
- Double DQN: online network chon action, target network danh gia action.

Uu diem:

- Giam overestimation bias.
- Thuong on dinh hon DQN.
- Phu hop khi reward nhieu nhieu va moi truong co tinh ngau nhien.

Nhuoc diem:

- Phuc tap hon DQN mot chut.
- Van can replay buffer, target network va tuning.

## 5. Bang so sanh

| Tieu chi | Q-learning | DQN | Double DQN |
|---|---|---|---|
| Kieu ham Q | Bang | Neural network | Neural network |
| Input | State roi rac | Vector/anh | Vector/anh |
| Kha nang mo rong | Thap | Cao | Cao |
| On dinh khi train | Cao voi state nho | Trung binh | Tot hon DQN |
| Overestimation bias | Co the co | Ro hon | Giam dang ke |
| Yeu cau tinh toan | Thap | Cao | Cao |
| Do kho cai dat | De | Trung binh | Trung binh/cao |
| Vai tro trong do an | Baseline | Deep RL baseline | Ban cai tien nen so sanh |

## 6. Ky vong ket qua tren map 15x15

Voi map 15x15:

1. Q-learning co the hoc nhanh va on dinh vi state space con nho.
2. DQN co the can nhieu episode hon de vuot Q-learning.
3. Double DQN thuong co reward trung binh on dinh hon DQN, nhat la khi ghost co target thong minh va moi truong kho hon random ghost.
4. Khi `ghost_chase_probability=1.0`, ghost se gan voi game goc hon, nen ket qua nen duoc doc kem `food_eaten` va `completion_rate`, khong chi nhin `win_rate`.

## 7. Cach trinh bay trong bao cao

Nen chia ket qua theo cac muc:

- Duong reward trung binh moi 50 hoac 100 episode.
- Ti le thang (win_rate).
- So buoc trung binh moi episode.
- So lan va cham ma.
- Nhan xet ve do on dinh cua training.

Ket luan ngan gon:

- Q-learning la baseline tot cho Pacman mini nho.
- DQN mo rong tot hon khi state phuc tap.
- Double DQN la cai tien cua DQN, thuong on dinh hon va giam viec danh gia Q-value qua cao.
