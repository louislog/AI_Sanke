# Snake Reinforcement Learning

[![Stars](https://img.shields.io/github/stars/louislog/AI_Sanke?style=flat-square)](https://github.com/louislog/AI_Sanke/stargazers)
[![Forks](https://img.shields.io/github/forks/louislog/AI_Sanke?style=flat-square)](https://github.com/louislog/AI_Sanke/network/members)
[![Issues](https://img.shields.io/github/issues/louislog/AI_Sanke?style=flat-square)](https://github.com/louislog/AI_Sanke/issues)
[![License](https://img.shields.io/github/license/louislog/AI_Sanke?style=flat-square)](#许可证)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-0f9d58?style=flat-square)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-ff6f00?style=flat-square)](https://stable-baselines3.readthedocs.io/)

用强化学习训练 AI 玩经典贪吃蛇。项目将 Pygame 实现的贪吃蛇游戏封装为 [Gymnasium](https://gymnasium.farama.org/) 标准环境，使用 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 的 PPO 算法进行策略优化，并提供训练、评估、可视化与手动游玩等完整工作流，适合作为 RL 入门与实践的参考项目。

## 功能特性

| 模块 | 说明 |
|------|------|
| **游戏核心** | `snake_game.py` 实现网格贪吃蛇逻辑与 Pygame 渲染，支持人机对战 |
| **RL 环境** | `SnakeEnv` 符合 Gymnasium API，10 维观测、3 离散动作、距离塑形奖励 |
| **PPO 训练** | 并行采样、定期 checkpoint、`EvalCallback` 自动保存最优模型 |
| **监控与日志** | TensorBoard 曲线、终端 `verbose` 统计、训练日志解读见下文 |
| **评估与导出** | `eval.py` 确定性推理，支持实时渲染与 MP4 视频导出 |
| **手动游玩** | 方向键 / WASD 控制，本地最高分记录 |

## 技术栈

- **算法**：PPO（Proximal Policy Optimization）
- **框架**：Gymnasium、Stable-Baselines3、PyTorch（SB3 依赖）
- **可视化**：Pygame、TensorBoard、imageio（视频导出）

## 快速开始

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 测试环境

```bash
python test_env.py
python test_env.py --render
```

### 训练

```bash
python train.py --n-envs 16 --total-timesteps 2000000 --save-freq 50000
```

检查点与 TensorBoard 日志保存在 `tmp/`。评估最优模型保存在 `tmp/best/best_model.zip`。

### TensorBoard

```bash
bash tensorboard.sh
```

### 评估

```bash
python eval.py --model tmp/best/best_model.zip
python eval.py --model tmp/best/best_model.zip --out-video gameplay.mp4
```

未指定 `--model` 时，自动使用 `tmp/` 下最新的检查点。

### 手动游玩

```bash
python snake_game.py
```

方向键 / WASD 控制，撞死后按 `R` 重开，`ESC` 退出。

## 项目结构

```
.
├── snake_game.py    # 游戏逻辑 + Pygame 渲染
├── snake_env.py     # Gymnasium 环境封装
├── train.py         # PPO 训练
├── eval.py          # 模型评估与视频导出
├── test_env.py      # 环境测试
├── runtime.py       # 运行时警告过滤
├── requirements.txt
└── tensorboard.sh
```

## RL 设计

### 观测（10 维特征）

前方/左侧/右侧危险、食物相对方向、当前方向 one-hot。

### 动作

- `0` 直行 | `1` 右转 | `2` 左转

### 奖励

吃到食物 +10，死亡 -20，每步 -0.001，靠近食物额外 +(距离缩短)×0.1。

## PPO 训练日志解读

`train.py` 设置 `verbose=1` 时，Stable-Baselines3 会周期性在终端打印训练统计。日志分为三组：**环境采样表现**、**训练进度**、**PPO 梯度更新**。

示例：

```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 497         |
|    ep_rew_mean          | 322         |
| time/                   |             |
|    fps                  | 18626       |
|    iterations           | 57          |
|    time_elapsed         | 200         |
|    total_timesteps      | 3735552     |
| train/                  |             |
|    approx_kl            | 0.001040139 |
|    clip_fraction        | 0.0118      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.154      |
|    explained_variance   | 0.263       |
|    learning_rate        | 0.0003      |
|    loss                 | 56.6        |
|    n_updates            | 560         |
|    policy_gradient_loss | 1.04e-05    |
|    value_loss           | 88.8        |
-----------------------------------------
```

### rollout/ — 环境采样表现

智能体在并行环境里实际玩游戏的表现（采样阶段，不是梯度更新那一步）。

| 指标 | 含义 |
|------|------|
| `ep_len_mean` | 最近一批 episode 的**平均步数**。一局从 `reset` 到死亡或达到 `max_steps` 截断。 |
| `ep_rew_mean` | 最近一批 episode 的**平均累计奖励**（含吃食物、死亡惩罚、每步惩罚、距离塑形）。 |

这两个指标最直观：**在奖励设计不变的前提下，越高通常越好**。本项目里 `ep_rew_mean` 大致可换算为「每局吃了多少食物」——例如 300 左右通常对应每局约 30 个食物（还需结合 `ep_len_mean` 看）。

> 注意：`rollout/` 统计来自训练时的**随机策略**采样；评估时请用 `eval.py`（`deterministic=True`）或 `tmp/best/best_model.zip`，两者可能不一致。

### time/ — 训练进度与速度

| 指标 | 含义 |
|------|------|
| `fps` | 每秒处理的**环境步数**（steps/s）。使用 `--n-envs 16` 等并行环境时数值会很大，反映采样吞吐，不是神经网络训练的 FPS。 |
| `iterations` | 已完成的 **PPO 迭代轮数**。每轮先采样 `n_steps × n_envs` 步，再对这批数据做多轮梯度更新。 |
| `time_elapsed` | 累计训练时间（秒）。 |
| `total_timesteps` | 累计**环境交互步数**（所有并行环境步数之和）。 |

### train/ — PPO 梯度更新

每轮采样结束后，用这批 rollout 数据更新策略网络（actor）和价值网络（critic）时的内部指标。

| 指标 | 含义 | 参考范围 / 说明 |
|------|------|-----------------|
| `learning_rate` | 当前学习率。 | 与 `train.py` 中设置一致（默认 `2.5e-4`）。 |
| `n_updates` | 累计**梯度更新次数**。 | 随训练单调增加。 |
| `clip_range` | PPO 裁剪超参数 ε。 | 固定超参（默认 `0.2`），限制策略单次更新幅度。 |
| `approx_kl` | 新旧策略的**近似 KL 散度**，衡量策略变化幅度。 | 太小（如 `< 0.01`）说明更新保守；过大（如 `> 0.03`）可能不稳定。 |
| `clip_fraction` | 触发 PPO ratio 裁剪的样本比例。 | 太低说明几乎没用到裁剪；过高说明更新经常撞到边界。 |
| `entropy_loss` | 策略熵相关项（SB3 以负号记录）。 | 绝对值越大，动作分布越随机、探索越多；趋近 0 表示策略更确定。 |
| `policy_gradient_loss` | **策略（actor）损失**。 | 单独看绝对值意义不大，关注趋势即可。 |
| `value_loss` | **价值（critic）损失**，预测回报与真实回报的误差。 | 偏大说明价值网络仍在学习；需结合 `explained_variance` 判断。 |
| `explained_variance` | 价值网络对回报的解释方差，范围约 0～1。 | 越接近 1 越好；长期偏低（如 `< 0.3`）说明 critic 拟合不足。 |
| `loss` | 总损失（策略 + 价值 + 熵等加权综合）。 | 主要看趋势，不宜单独解读绝对值。 |

### 快速对照

| 想看什么 | 主要看 |
|----------|--------|
| 蛇玩得怎么样 | `rollout/ep_rew_mean`、`rollout/ep_len_mean` |
| 训练到哪了 | `time/total_timesteps`、`time/iterations` |
| 更新是否稳定 | `train/approx_kl`、`train/clip_fraction` |
| 探索 vs 利用 | `train/entropy_loss` |
| 价值网络是否学好 | `train/explained_variance`、`train/value_loss` |

TensorBoard 中同名指标可在 `bash tensorboard.sh` 启动后查看曲线变化；选模型时建议以 `EvalCallback` 写入的 `tmp/best/best_model.zip` 为准，而非仅看最后一轮 `rollout` 数值。

## 参考

- [helicopter-rl](https://github.com/rossning92/helicopter-rl)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)

## Star 趋势

[![Star History Chart](https://api.star-history.com/svg?repos=louislog/AI_Sanke&type=Date)](https://star-history.com/#louislog/AI_Sanke&Date)

## 许可证

MIT
