"""
Main entry point — tests the environment works correctly
Run: python main.py
"""

import numpy as np
from src.environment_gymnasium import BugTriageGymnasiumEnv


def test_environment(task_level: str = "medium"):
    print(f"\n{'='*50}")
    print(f"Testing Bug Triage RL Environment — {task_level.upper()}")
    print(f"{'='*50}")

    env = BugTriageGymnasiumEnv(task_level=task_level, max_episode_steps=20)

    obs, info = env.reset(seed=42)
    print(f"Reset successful!")
    print(f"Task level: {info['task_level']}")
    print(f"Team size: {info['team_size']}")
    print(f"Total bugs: {info['total_bugs']}")
    print(f"Observation shape: {obs.shape}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {step+1}: action={action} | reward={reward:.3f} | cumulative={total_reward:.3f}")
        if 'action_type' in info:
            print(f"  Action: {info['action_type']} on {info.get('bug_id', 'N/A')} ({info.get('bug_severity', 'N/A')})")
            print(f"  Reason: {info.get('reward_reason', 'N/A')}")

        if terminated or truncated:
            print("\nEpisode ended!")
            break

    env.render()

    state = env.get_state_dict()
    print(f"\nFinal state:")
    print(f"  Queue health: {state['queue_health']}%")
    print(f"  Team availability: {state['team_availability']}%")
    print(f"  Open bugs: {state['open_bugs']}")
    print(f"  Total reward: {state['cumulative_reward']}")

    print(f"\nEnvironment test PASSED for {task_level}!")
    return True


if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        try:
            test_environment(level)
        except Exception as e:
            print(f"FAILED for {level}: {e}")
            import traceback
            traceback.print_exc()