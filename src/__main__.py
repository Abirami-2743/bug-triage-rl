"""
Main entry point for the Bug Triage RL Environment
Provides CLI interface for testing and evaluation.
"""

import argparse
import sys
import os
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.environment import BugTriageEnv
from src.models import Action, ActionType
from src.tasks import TaskManager
from src.bug_generator import BugGenerator


def run_random_agent(task_level: str, max_steps: int, render: bool = False):
    """Run a random agent for testing"""
    import random
    
    env = BugTriageEnv(task_level=task_level, max_episode_steps=max_steps)
    obs = env.reset()
    
    total_reward = 0.0
    step_count = 0
    done = False
    
    print(f"Starting random agent on {task_level} task...")
    
    while not done and step_count < max_steps:
        # Random action selection
        available_actions = env.get_action_space()
        action_type = random.choice(available_actions)
        
        # Get random bug
        if env.bug_queue.bugs:
            bug = random.choice(env.bug_queue.bugs)
            bug_id = bug.id
        else:
            break
        
        # Get random developer if assigning
        developer_id = None
        if action_type.value == "assign":
            available_devs = [dev for dev in env.team_state.developers if dev.is_available]
            if available_devs:
                developer_id = random.choice(available_devs).id
            else:
                # No available developers, choose a different action
                action_type = random.choice([ActionType.ESCALATE, ActionType.DEFER, ActionType.CLOSE])
        
        # Create action
        action = Action(
            action_type=action_type,
            bug_id=bug_id,
            developer_id=developer_id if action_type == ActionType.ASSIGN else None,
            reasoning="Random agent action"
        )
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if render:
            env.render()
            print(f"Step {step_count}: {action_type.value} -> {bug_id}, Reward: {reward:.2f}")
    
    # Grade episode
    task_manager = TaskManager()
    final_score = task_manager.grade_episode(task_level, env, env.episode_metrics)
    
    print(f"\nEpisode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final score: {final_score:.2f}")
    print(f"Bugs processed: {env.episode_metrics.total_bugs_processed}")
    print(f"SLA compliance: {env.episode_metrics.bugs_resolved_within_sla}/{env.episode_metrics.total_bugs_processed}")
    
    return final_score


def evaluate_baseline(task_level: str, num_episodes: int = 10):
    """Evaluate baseline random agent performance"""
    task_manager = TaskManager()
    
    print(f"Evaluating baseline performance on {task_level} task...")
    print(f"Running {num_episodes} episodes...")
    
    scores = task_manager.evaluate_baseline_performance(task_level, num_episodes)
    
    print(f"\nBaseline Performance Results:")
    print(f"Mean score: {scores['mean_score']:.3f}")
    print(f"Min score: {scores['min_score']:.3f}")
    print(f"Max score: {scores['max_score']:.3f}")
    print(f"Std deviation: {scores['std_score']:.3f}")
    
    return scores


def test_environment(task_level: str):
    """Test environment functionality"""
    print(f"Testing {task_level} environment...")
    
    # Test initialization
    env = BugTriageEnv(task_level=task_level)
    print(f"✅ Environment initialized successfully")
    
    # Test reset
    obs = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   - Bugs in queue: {len(env.bug_queue.bugs)}")
    print(f"   - Team size: {len(env.team_state.developers)}")
    print(f"   - Queue health: {env.bug_queue.queue_health_score:.2f}")
    
    # Test step
    if env.bug_queue.bugs and env.team_state.developers:
        bug = env.bug_queue.bugs[0]
        dev = env.team_state.developers[0]
        
        action = Action(
            action_type=ActionType.ASSIGN,
            bug_id=bug.id,
            developer_id=dev.id,
            reasoning="Test action"
        )
        
        obs, reward, done, info = env.step(action)
        print(f"✅ Step executed successfully")
        print(f"   - Action: {action.action_type.value} {bug.id} to {dev.id}")
        print(f"   - Reward: {reward:.2f}")
        print(f"   - Done: {done}")
    
    # Test state
    state = env.state()
    print(f"✅ State retrieval successful")
    print(f"   - Current step: {state.observation.current_step}")
    print(f"   - Total steps: {state.observation.total_steps}")
    
    # Test task grader
    task_manager = TaskManager()
    task_info = task_manager.get_task_info(task_level)
    print(f"✅ Task grader working")
    print(f"   - Passing score: {task_info['success_criteria']['passing_score']}")
    
    print(f"\n✅ All tests passed for {task_level} environment!")


def show_task_info(task_level: str):
    """Show detailed task information"""
    task_manager = TaskManager()
    task_info = task_manager.get_task_info(task_level)
    
    print(f"\n{'='*60}")
    print(f"TASK INFORMATION: {task_level.upper()}")
    print(f"{'='*60}")
    
    criteria = task_info['success_criteria']
    
    print(f"\nDescription: {criteria['description']}")
    print(f"\nRequirements:")
    for i, req in enumerate(criteria['requirements'], 1):
        print(f"  {i}. {req}")
    
    print(f"\nPassing Score: {criteria['passing_score']*100:.0f}%")
    
    print(f"\nScore Weighting:")
    for component, weight in criteria['weighting'].items():
        print(f"  - {component.replace('_', ' ').title()}: {weight*100:.0f}%")
    
    print(f"\n{'='*60}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Bug Triage RL Environment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src --task easy --test
  python -m src --task medium --random-agent --render
  python -m src --task hard --baseline --episodes 20
  python -m src --task medium --info
        """
    )
    
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"],
                       help="Task difficulty level")
    
    parser.add_argument("--test", action="store_true",
                       help="Test environment functionality")
    
    parser.add_argument("--random-agent", action="store_true",
                       help="Run random agent for testing")
    
    parser.add_argument("--baseline", action="store_true",
                       help="Evaluate baseline performance")
    
    parser.add_argument("--info", action="store_true",
                       help="Show task information")
    
    parser.add_argument("--render", action="store_true",
                       help="Render environment state")
    
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for baseline evaluation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if sum([args.test, args.random_agent, args.baseline, args.info]) == 0:
        parser.error("Must specify one of: --test, --random-agent, --baseline, --info")
    
    try:
        if args.info:
            show_task_info(args.task)
        
        elif args.test:
            test_environment(args.task)
        
        elif args.random_agent:
            run_random_agent(args.task, args.max_steps, args.render)
        
        elif args.baseline:
            evaluate_baseline(args.task, args.episodes)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
