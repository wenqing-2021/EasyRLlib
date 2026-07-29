# ruff: noqa: I001
"""Create headless training plots and rendered evaluations for EasyRLlib runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/easyrllib-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _safe_artifact_name(run_directory: Path, output_directory: Path) -> str:
    relative_name = run_directory.relative_to(output_directory).as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", relative_name)


def _read_progress(progress_path: Path) -> list[dict[str, float]]:
    with progress_path.open(newline="") as progress_file:
        reader = csv.DictReader(progress_file, delimiter="\t")
        return [
            {key: float(value) for key, value in row.items() if value not in (None, "")}
            for row in reader
        ]


def plot_training_rewards(output_directory: Path, vis_directory: Path) -> list[Path]:
    """Plot average episode return for every EasyRLlib progress file."""
    output_directory = output_directory.resolve()
    vis_directory = vis_directory.resolve()
    vis_directory.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    for progress_path in sorted(output_directory.rglob("progress.txt")):
        rows = _read_progress(progress_path)
        if not rows or "Epoch" not in rows[0] or "AverageEpRet" not in rows[0]:
            continue

        epochs = np.asarray([row["Epoch"] for row in rows])
        average_returns = np.asarray([row["AverageEpRet"] for row in rows])
        run_directory = progress_path.parent
        figure, axis = plt.subplots(figsize=(9, 5), dpi=160)
        axis.plot(epochs, average_returns, color="#1f77b4", linewidth=1.8, label="AverageEpRet")

        if "MinEpRet" in rows[0] and "MaxEpRet" in rows[0]:
            minimum_returns = np.asarray([row["MinEpRet"] for row in rows])
            maximum_returns = np.asarray([row["MaxEpRet"] for row in rows])
            axis.fill_between(
                epochs,
                minimum_returns,
                maximum_returns,
                color="#1f77b4",
                alpha=0.16,
                label="Min/Max EpRet",
            )

        axis.set_title(run_directory.parent.name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Episode return")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()

        artifact_path = vis_directory / (
            f"{_safe_artifact_name(run_directory, output_directory)}_reward_curve.png"
        )
        figure.savefig(artifact_path)
        plt.close(figure)
        generated_plots.append(artifact_path)

    summary_path = vis_directory / "training_reward_curves.json"
    summary_path.write_text(
        json.dumps([str(path.relative_to(REPOSITORY_ROOT)) for path in generated_plots], indent=2)
        + "\n"
    )
    return generated_plots


def evaluate_dueling_dqn(
    config_path: Path,
    checkpoint_path: Path,
    vis_directory: Path,
    episodes: int,
) -> tuple[Path, Path]:
    """Evaluate a Dueling DQN checkpoint and save a GIF for its first episode."""
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    vis_directory = vis_directory.resolve()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import gymnasium as gym
    import torch

    from src.agent_lib.dqn import DuelingDQN
    from src.config.configure import load_config

    run_config = load_config(str(config_path))
    environment = gym.make(run_config.env_config.env_name, render_mode="rgb_array")
    agent = DuelingDQN(
        environment.observation_space, environment.action_space, run_config
    )
    state_dict = torch.load(
        checkpoint_path, map_location=agent.device, weights_only=True
    )
    agent.load_state_dict(state_dict)
    agent.policy.eval()
    agent.epsilon = 0.0

    episode_returns = []
    frames = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=run_config.train_config.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            if episode == 0:
                frames.append(Image.fromarray(environment.render()))
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward
        episode_returns.append(float(episode_return))
    environment.close()

    if not frames:
        raise RuntimeError("The evaluation produced no render frames")

    vis_directory.mkdir(parents=True, exist_ok=True)
    gif_path = vis_directory / "DuelingDQN_CartPole_eval.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=33,
        loop=0,
        optimize=False,
    )
    metrics_path = vis_directory / "DuelingDQN_CartPole_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                "episodes": episodes,
                "episode_returns": episode_returns,
                "mean_return": float(np.mean(episode_returns)),
                "std_return": float(np.std(episode_returns)),
                "deterministic": True,
            },
            indent=2,
        )
        + "\n"
    )
    return gif_path, metrics_path


def evaluate_ppo_bipedalwalker(
    config_path: Path,
    checkpoint_path: Path,
    vis_directory: Path,
    episodes: int,
) -> tuple[Path, Path]:
    """Evaluate a PPO BipedalWalker checkpoint and save a deterministic GIF."""
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    vis_directory = vis_directory.resolve()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import gymnasium as gym
    import torch

    from src.agent_lib.ppo import PPO
    from src.config.configure import load_config

    run_config = load_config(str(config_path))
    environment = gym.make(run_config.env_config.env_name, render_mode="rgb_array")
    agent = PPO(environment.observation_space, environment.action_space, run_config)
    state_dict = torch.load(
        checkpoint_path, map_location=agent.device, weights_only=True
    )
    agent.load_state_dict(state_dict)
    agent.policy.eval()
    agent.critic.eval()

    episode_returns = []
    frames = []
    frame_stride = 4
    for episode in range(episodes):
        observation, _ = environment.reset(seed=run_config.train_config.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        step = 0
        while not (terminated or truncated):
            if episode == 0 and step % frame_stride == 0:
                frames.append(Image.fromarray(environment.render()))
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=agent.device
            )
            with torch.no_grad():
                action = agent.policy._distribution(observation_tensor).mean
            action = action.cpu().numpy()
            action = np.clip(action, environment.action_space.low, environment.action_space.high)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward
            step += 1
        episode_returns.append(float(episode_return))
    environment.close()

    if not frames:
        raise RuntimeError("The evaluation produced no render frames")

    vis_directory.mkdir(parents=True, exist_ok=True)
    gif_path = vis_directory / "PPO_BipedalWalker_eval.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=33,
        loop=0,
        optimize=False,
    )
    metrics_path = vis_directory / "PPO_BipedalWalker_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                "episodes": episodes,
                "episode_returns": episode_returns,
                "mean_return": float(np.mean(episode_returns)),
                "std_return": float(np.std(episode_returns)),
                "deterministic": True,
                "render_frame_stride": frame_stride,
            },
            indent=2,
        )
        + "\n"
    )
    return gif_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves_parser = subparsers.add_parser("curves", help="Plot all output rewards")
    curves_parser.add_argument("--output-dir", type=Path, default=Path("output"))
    curves_parser.add_argument("--vis-dir", type=Path, default=Path("vis"))

    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Evaluate Dueling DQN and render one episode"
    )
    evaluation_parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment/dueling_dqn_cartpole/run_config.yaml"),
    )
    evaluation_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "output/DuelingDQN_CartPole/DuelingDQN_CartPole_s1/pyt_save/model225.pt"
        ),
    )
    evaluation_parser.add_argument("--vis-dir", type=Path, default=Path("vis"))
    evaluation_parser.add_argument("--episodes", type=int, default=10)

    ppo_evaluation_parser = subparsers.add_parser(
        "evaluate-ppo", help="Evaluate PPO BipedalWalker and render one episode"
    )
    ppo_evaluation_parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment/ppo_BipedalWalker-v3/run_config.yaml"),
    )
    ppo_evaluation_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "output/PPO_BipedalWalker/PPO_BipedalWalker_s1/pyt_save/model249.pt"
        ),
    )
    ppo_evaluation_parser.add_argument("--vis-dir", type=Path, default=Path("vis"))
    ppo_evaluation_parser.add_argument("--episodes", type=int, default=10)

    arguments = parser.parse_args()
    if arguments.command == "curves":
        plots = plot_training_rewards(arguments.output_dir, arguments.vis_dir)
        print(f"Generated {len(plots)} reward curves in {arguments.vis_dir}")
    elif arguments.command == "evaluate":
        gif_path, metrics_path = evaluate_dueling_dqn(
            arguments.config,
            arguments.checkpoint,
            arguments.vis_dir,
            arguments.episodes,
        )
        print(f"Saved GIF to {gif_path}")
        print(f"Saved evaluation metrics to {metrics_path}")
    else:
        gif_path, metrics_path = evaluate_ppo_bipedalwalker(
            arguments.config,
            arguments.checkpoint,
            arguments.vis_dir,
            arguments.episodes,
        )
        print(f"Saved GIF to {gif_path}")
        print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
