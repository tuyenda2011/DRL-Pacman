$ErrorActionPreference = "Stop"

& python -m src.training.train_q_learning --config configs/q_learning/q_learning_lr_0001.yaml
& python -m src.training.train_dqn --config configs/dqn/dqn_lr_0001.yaml
& python -m src.training.train_double_dqn --config configs/double_dqn/double_dqn_lr_0001.yaml
