import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.pick_place_env import SO101PickPlaceEnv


def make_env(rank: int, max_steps: int = 250):
    def _init():
        env = SO101PickPlaceEnv(observation_mode="state", max_steps=max_steps)
        env = Monitor(env)
        return env
    return _init


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Vectorized environments for fast sample collection
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.max_steps) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.max_steps)])

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Evaluation environment
    eval_env = DummyVecEnv([make_env(999, args.max_steps)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_so101",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best_model"),
        log_path=args.log_dir,
        eval_freq=max(5000 // args.num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=args.log_dir,
        device="cuda" if args.use_cuda else "cpu",
    )

    print(f"Starting PPO training on {args.num_envs} environments for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO_SO101",
    )

    final_model_path = os.path.join(args.checkpoint_dir, "ppo_so101_final.zip")
    norm_path = os.path.join(args.checkpoint_dir, "vecnormalize.pkl")
    model.save(final_model_path)
    env.save(norm_path)
    print(f"Training complete! Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for SO-101 Pick and Place")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--log-dir", type=str, default="logs/ppo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--use-cuda", action="store_true", default=True)
    args = parser.parse_args()
    train(args)
