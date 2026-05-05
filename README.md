# LMMARL: Joint Task Offloading and Resource Allocation in Multi-UAV Collaborative Computing

[![Paper](https://img.shields.io/badge/Paper-Ad%20Hoc%20Networks-blue)](https://www.sciencedirect.com/science/article/pii/S157087052600154X) [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.adhoc.2026.104288-green)](https://doi.org/10.1016/j.adhoc.2026.104288)

## 项目概述

LMMARL（Leader-Member Cooperative Multi-Agent Reinforcement Learning，领导者-成员协作多智能体强化学习）是一个面向无人机辅助雾计算系统的分层多智能体强化学习框架，用于联合优化任务卸载与资源调度。该框架通过自适应部分卸载、跨层路径选择和无人机集群动态协作，协调客户端-边缘-雾三层决策，以最小化系统总时延。

## 主要特性

- 客户端层基于 PPO 的自适应部分卸载
- 领导者无人机 DQN+PPO 双模块架构：跨层路径选择与精细化任务分配
- 成员无人机独立 DQN 智能体动态参与决策
- 支持同构与异构无人机计算环境
- 内置消融模式：`proposed`、`binary_offloading`、`no_fog`、`no_collaboration`

## 环境要求

```
python >= 3.8
torch >= 1.10.0
numpy >= 1.21.0
```

安装依赖：

```bash
pip install -r requirements.txt
```

> 如有可用 GPU，代码会自动检测并启用加速。如需安装 CUDA 版 PyTorch，请参见 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

---

## 项目结构

```
LMMARL/
├── __init__.py              # 包初始化
├── rl_agents.py             # DQN 与 PPO 智能体实现
├── client_layer.py          # 客户端层：任务车辆定义
├── edge_layer.py            # 边缘层：领导者/成员无人机定义
├── fog_layer.py             # 雾计算层：地面基站定义
├── communication.py         # 通信模型（信道增益、LoS/NLoS、传输速率与时延）
├── computational.py         # 计算模型（各层执行时延）
├── main_simulation.py       # 主仿真入口：训练循环与评估
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 快速开始

本项目使用相对导入，请在 `LMMARL/` 的 **上级目录** 下运行。

```python
from LMMARL.main_simulation import run_single_simulation

avg_delay, avg_compute_delay, rewards = run_single_simulation(
    num_vehicles=40,       # 任务车辆数量 (K)
    num_uavs=10,           # 成员无人机数量 (U)
    w1=0.5,                # 本地时延权重
    w2=0.5,                # 卸载时延权重
    episodes=500,          # 训练轮数
    steps_per_episode=200, # 每轮步数
    debug=True             # 打印训练过程
)

print(f"平均系统时延: {avg_delay:.4f} s")
print(f"平均计算时延: {avg_compute_delay:.4f} s")
```

---

## 参数说明

### `run_single_simulation()`

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `num_vehicles` | int | 任务车辆数量 (K) | 必填 |
| `num_uavs` | int | 成员无人机数量 (U) | 必填 |
| `w1` | float | 本地执行时延权重 | 必填 |
| `w2` | float | 卸载时延权重 | 必填 |
| `episodes` | int | 训练总轮数 | 500 |
| `steps_per_episode` | int | 每轮仿真步数 | 200 |
| `mode` | str | 运行模式（见下表） | `'proposed'` |
| `dqn_lr` | float | DQN 学习率 | 5e-5 |
| `ppo_lr` | float | PPO 学习率 | 5e-5 |
| `gamma` | float | 折扣因子 | 0.9 |
| `task_size_range` | tuple | 单任务数据量范围（MB） | (1.0, 3.0) |
| `tasks_per_slot_range` | tuple | 每时隙每车辆生成任务数 | (5, 10) |
| `random_seed` | int | 随机种子 | 42 |
| `debug` | bool | 是否打印训练日志 | False |

### 运行模式

| 模式 | 说明 |
|------|------|
| `proposed` | 完整 LMMARL 框架（默认） |
| `binary_offloading` | 消融：基于阈值的二元卸载，客户端不使用 RL |
| `no_fog` | 消融：禁用雾计算路径 |
| `no_collaboration` | 消融：仅选择最优单架无人机，无集群协作 |

### 返回值

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `avg_delay` | float | 稳定阶段（最后 10% 轮次）平均系统总时延（秒） |
| `avg_compute_delay` | float | 稳定阶段平均计算时延（秒） |
| `rewards_history` | list | 每轮平均奖励值列表，可用于绘制收敛曲线 |

---

## 默认仿真参数

| 参数 | 值 |
|------|-----|
| 仿真区域 | 500 m × 500 m |
| 无人机飞行高度 (H) | 100 m |
| 地面基站数量 | 3 |
| 信道带宽 (B) | 1 MHz |
| 领导者无人机发射功率 | 2 W |
| 车辆发射功率 | 1 W |
| 噪声功率 | −120 dBm |
| 参考信道增益 (g₀) | −60 dB |
| LoS 概率参数 (a, b) | 10, 0.6 |
| NLoS 衰减系数 (η) | 0.2 |
| 车辆计算能力 (f_k) | 2 GHz |
| 成员无人机计算能力 (f_i) | 10 GHz |
| 基站计算能力 (f_j) | 25 GHz |
| 任务复杂度 | 500–1500 CPU cycles/bit |
| DQN 经验回放缓冲区 | 10,000 |
| 小批量大小 | 64 |
| PPO 裁剪系数 | 0.2 |
| PPO 更新轮次 | 10 |

---

## 论文

本项目对应论文已发表于 **Ad Hoc Networks**（Elsevier）：

> Weiyi Wang, Zhengshu Zhou, Li Zhao, Qiang Zhi. "Joint task offloading and resource allocation in multi-UAV collaborative computing via LMMARL," *Ad Hoc Networks*, vol. 190, pp. 104288, 2026.

**[📄 在线阅读（ScienceDirect）](https://www.sciencedirect.com/science/article/pii/S157087052600154X)**

## 引用

如果本代码对您的研究有帮助，请引用：

```bibtex
@article{WANG2026104288,
  title     = {Joint task offloading and resource allocation in multi-UAV collaborative computing via LMMARL},
  author    = {Weiyi Wang and Zhengshu Zhou and Li Zhao and Qiang Zhi},
  journal   = {Ad Hoc Networks},
  volume    = {190},
  pages     = {104288},
  year      = {2026},
  issn      = {1570-8705},
  doi       = {https://doi.org/10.1016/j.adhoc.2026.104288},
  url       = {https://www.sciencedirect.com/science/article/pii/S157087052600154X},
  keywords  = {Unmanned Aerial Vehicles, Fog computing, Task offloading, Resource scheduling, Multi-Agent Reinforcement Learning}
}
```

## 许可

本代码仅供学术研究使用。
