import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pick_place_env import SO101PickPlaceEnv


def evaluate(args):
    def make_env():
        return SO101PickPlaceEnv(observation_mode="state", max_steps=args.max_steps)

    env = DummyVecEnv([make_env])
    if os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded normalizer from {args.vecnormalize_path}")

    model = PPO.load(args.model_path, device="cpu")
    print(f"Loaded trained PPO policy from {args.model_path}")

    successes = 0
    grasps = 0
    lifts = 0

    print(f"\n--- Running {args.episodes} Evaluation Episodes ---")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            ep_reward += reward[0]
            done = done_arr[0]

        inf = info[0]
        if inf["is_grasped"]:
            grasps += 1
        if inf["is_lifted"]:
            lifts += 1
        if inf["is_success"]:
            successes += 1

        print(
            f"Ep {ep+1:2d}: Reward={ep_reward:6.2f} | Grasped={inf['is_grasped']} | "
            f"Lifted={inf['is_lifted']} | Success={inf['is_success']}"
        )

    print("\n" + "=" * 45)
    print(f"Grasp Rate:   {grasps / args.episodes * 100:.1f}%")
    print(f"Lift Rate:    {lifts / args.episodes * 100:.1f}%")
    print(f"Success Rate: {successes / args.episodes * 100:.1f}%")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SO-101 policy")
    parser.add_argument("--model-path", type=str, default="checkpoints/ppo_so101_final.zip")
    parser.add_argument("--vecnormalize-path", type=str, default="checkpoints/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()
    evaluate(args)
