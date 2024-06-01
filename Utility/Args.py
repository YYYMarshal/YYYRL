from dataclasses import dataclass
import torch


@dataclass
class Args:
    num_episodes: int = 500
    seed: int = 0
    # learning rate
    lr: int = 2e-3
    hidden_dim: int = 128
    gamma: float = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ====== Valued-based Method ======
    epsilon: float = 0.01

    # ====== Off-policy Method ======
    # 经过 target_update 个 step 后目标网络进行更新
    target_update: int = 10
    buffer_size: int = 10000
    # 当 buffer 数据的数量超过 minimal_size 后,才进行Q网络训练
    minimal_size: int = 500
    batch_size: int = 64
