"""
Gymnasium-Compliant Bug Triage RL Environment
Follows OpenEnv specification with proper Gymnasium API
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from .models import (
    Bug, BugQueue, TeamState, Developer, DeveloperSkill, BugType,
    Action, Reward, ActionType, BugStatus, BugSeverity, EpisodeMetrics
)
from .bug_generator import BugGenerator
from .reward_function import RewardCalculator


class BugTriageGymnasiumEnv(gym.Env):
    """
    Gymnasium-compliant Bug Triage Environment
    Follows standard Gymnasium API for OpenEnv compatibility
    """

    metadata = {
        'render_modes': ['human'],
        'render_fps': 4
    }

    def __init__(
        self,
        task_level: str = "medium",
        max_episode_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.task_level = task_level
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.bug_generator = BugGenerator(task_level)
        self.reward_calculator = RewardCalculator(task_level)

        self.current_step = 0
        self.bug_queue: BugQueue = BugQueue(bugs=[])
        self.team_state: TeamState = TeamState(developers=[])
        self.cumulative_reward = 0.0
        self.episode_metrics: EpisodeMetrics = EpisodeMetrics()
        self.initial_queue_health = 0.0
        self.action_history: List[Dict] = []

        self._setup_team()
        self._define_spaces()

    def _define_spaces(self):
        """Define flat observation and action spaces for Gymnasium compatibility"""

        # Flat action space: 4 action types × 50 bugs × 8 developers = combined
        # Using MultiDiscrete for simplicity
        self.action_space = spaces.MultiDiscrete([4, 50, 8])

        # Flat observation space — normalized floats
        # [queue_health, team_availability, critical_count_norm, avg_age_norm,
        #  step_norm, bug0_severity, bug0_priority, bug0_sla_ratio, ... x10 bugs]
        obs_size = 5 + (10 * 4)  # 5 global + 10 bugs x 4 features each
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

    def _setup_team(self):
        if self.task_level == "easy":
            team_config = [
                ("dev-001", "Alice", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Bob", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol", [DeveloperSkill.FULLSTACK])
            ]
        elif self.task_level == "medium":
            team_config = [
                ("dev-001", "Sarah Chen", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Marcus Johnson", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol White", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David Kim", [DeveloperSkill.FULLSTACK])
            ]
        else:  # hard
            team_config = [
                ("dev-001", "Sarah Chen", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Marcus Johnson", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol White", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David Kim", [DeveloperSkill.FULLSTACK]),
                ("dev-005", "Eve Torres", [DeveloperSkill.SECURITY, DeveloperSkill.BACKEND]),
                ("dev-006", "Frank Lee", [DeveloperSkill.DEVOPS, DeveloperSkill.DATABASE]),
                ("dev-007", "Grace Park", [DeveloperSkill.FRONTEND, DeveloperSkill.FULLSTACK]),
                ("dev-008", "Henry Adams", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE])
            ]

        developers = []
        for dev_id, name, skills in team_config:
            developer = Developer(
                id=dev_id,
                name=name,
                skills=skills,
                max_capacity=3
            )
            developers.append(developer)

        self.team_state = TeamState(developers=developers)
        self.team_state.update_metrics()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.cumulative_reward = 0.0
        self.action_history = []

        initial_bugs = self.bug_generator.generate_bug_batch()
        self.bug_queue = BugQueue(bugs=initial_bugs)
        self.bug_queue.update_metrics()

        for dev in self.team_state.developers:
            dev.current_load = 0
            dev.availability = True
        self.team_state.update_metrics()

        self.episode_metrics = EpisodeMetrics()
        self.initial_queue_health = self.bug_queue.queue_health_score

        observation = self._get_flat_observation()
        info = {
            'task_level': self.task_level,
            'initial_queue_health': self.initial_queue_health,
            'team_size': len(self.team_state.developers),
            'total_bugs': len(self.bug_queue.bugs)
        }

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        try:
            action_type_idx = int(action[0])
            bug_idx = int(action[1])
            dev_idx = int(action[2])

            action_type_map = {
                0: ActionType.ASSIGN,
                1: ActionType.ESCALATE,
                2: ActionType.DEFER,
                3: ActionType.CLOSE
            }
            action_type = action_type_map[action_type_idx]

            # Get open bugs
            open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
            if not open_bugs:
                # No open bugs — episode should end
                observation = self._get_flat_observation()
                self.current_step += 1
                terminated = True
                truncated = self.current_step >= self.max_episode_steps
                return observation, 0.0, terminated, truncated, {'no_open_bugs': True}

            # Pick bug by index (wrap around)
            bug = open_bugs[bug_idx % len(open_bugs)]

            # Pick developer
            developer_id = None
            if action_type == ActionType.ASSIGN:
                available_devs = [d for d in self.team_state.developers if d.is_available]
                if available_devs:
                    developer_id = available_devs[dev_idx % len(available_devs)].id
                else:
                    # No available devs — penalize
                    self.current_step += 1
                    observation = self._get_flat_observation()
                    terminated = False
                    truncated = self.current_step >= self.max_episode_steps
                    return observation, -0.3, terminated, truncated, {'no_available_devs': True}

            action_obj = Action(
                action_type=action_type,
                bug_id=bug.id,
                developer_id=developer_id,
                reasoning="Agent action"
            )

        except Exception as e:
            self.current_step += 1
            observation = self._get_flat_observation()
            return observation, -0.5, False, self.current_step >= self.max_episode_steps, {'error': str(e)}

        # Calculate and apply reward
        previous_health = self.bug_queue.queue_health_score
        reward_obj = self.reward_calculator.calculate_reward(
            action=action_obj.action_type,
            bug=bug,
            team_state=self.team_state,
            bug_queue=self.bug_queue,
            previous_queue_health=previous_health
        )

        self._apply_action(action_obj, bug)
        self.cumulative_reward += reward_obj.immediate_reward
        self.current_step += 1

        # In hard mode, stream new bugs occasionally
        if self.task_level == "hard" and self.current_step % 10 == 0:
            new_bug = self.bug_generator.generate_streaming_bug(self.current_step)
            self.bug_queue.bugs.append(new_bug)
            self.bug_queue.update_metrics()

        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps

        observation = self._get_flat_observation()
        info = {
            'action_type': action_obj.action_type.value,
            'bug_id': bug.id,
            'bug_severity': bug.severity.value,
            'reward_components': reward_obj.reward_components,
            'reward_reason': reward_obj.reason,
            'queue_health': self.bug_queue.queue_health_score,
            'team_availability': self.team_state.availability_ratio,
            'cumulative_reward': self.cumulative_reward,
            'step': self.current_step
        }

        self.action_history.append({
            'step': self.current_step,
            'action_type': action_obj.action_type.value,
            'bug_id': bug.id,
            'reward': reward_obj.immediate_reward
        })

        return observation, reward_obj.immediate_reward, terminated, truncated, info

    def _apply_action(self, action: Action, bug: Bug):
        if action.action_type == ActionType.ASSIGN and action.developer_id:
            developer = next(
                (d for d in self.team_state.developers if d.id == action.developer_id), None
            )
            if developer:
                bug.status = BugStatus.IN_PROGRESS
                bug.assigned_to = action.developer_id
                developer.current_load += 1
                if developer.current_load >= developer.max_capacity:
                    developer.availability = False

        elif action.action_type == ActionType.ESCALATE:
            bug.status = BugStatus.ESCALATED
            bug.escalation_level += 1
            self.episode_metrics.total_bugs_processed += 1

        elif action.action_type == ActionType.DEFER:
            bug.status = BugStatus.DEFERRED
            bug.age_hours += 2  # Penalty: bug gets older

        elif action.action_type == ActionType.CLOSE:
            bug.status = BugStatus.CLOSED
            self.episode_metrics.total_bugs_processed += 1
            if bug.age_hours <= bug.sla_hours:
                self.episode_metrics.bugs_resolved_within_sla += 1

        self.bug_queue.update_metrics()
        self.team_state.update_metrics()

    def _check_termination(self) -> bool:
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        return len(open_bugs) == 0

    def _get_flat_observation(self) -> np.ndarray:
        """Return flat normalized observation array"""
        obs = []

        # Global metrics (5 features)
        obs.append(float(self.bug_queue.queue_health_score))
        obs.append(float(self.team_state.availability_ratio))
        obs.append(float(min(self.bug_queue.critical_count / 10.0, 1.0)))
        obs.append(float(min(self.bug_queue.avg_age_hours / 168.0, 1.0)))
        obs.append(float(self.current_step / self.max_episode_steps))

        # Top 10 bugs features (4 features each = 40)
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        # Sort by priority
        open_bugs.sort(key=lambda b: b.priority_score, reverse=True)

        severity_map = {
            BugSeverity.LOW: 0.25,
            BugSeverity.MEDIUM: 0.5,
            BugSeverity.HIGH: 0.75,
            BugSeverity.CRITICAL: 1.0
        }

        for i in range(10):
            if i < len(open_bugs):
                bug = open_bugs[i]
                obs.append(severity_map[bug.severity])
                obs.append(float(bug.priority_score))
                obs.append(float(min(bug.age_hours / bug.sla_hours, 1.0)))  # SLA usage ratio
                obs.append(float(min(bug.affected_users / 5000.0, 1.0)))
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def get_state_dict(self) -> Dict:
        """Returns full state as dict — used by FastAPI"""
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        bugs_data = []
        for bug in sorted(open_bugs, key=lambda b: b.priority_score, reverse=True):
            bugs_data.append({
                'id': bug.id,
                'title': bug.title,
                'severity': bug.severity.value,
                'bug_type': bug.bug_type.value,
                'status': bug.status.value,
                'age_hours': bug.age_hours,
                'affected_users': bug.affected_users,
                'priority_score': round(bug.priority_score, 2),
                'sla_hours': bug.sla_hours,
                'sla_usage': round(min(bug.age_hours / bug.sla_hours * 100, 150), 1)
            })

        devs_data = []
        for dev in self.team_state.developers:
            devs_data.append({
                'id': dev.id,
                'name': dev.name,
                'skills': [s.value for s in dev.skills],
                'current_load': dev.current_load,
                'max_capacity': dev.max_capacity,
                'is_available': dev.is_available,
                'utilization': round(dev.current_load / dev.max_capacity * 100, 1)
            })

        return {
            'task_level': self.task_level,
            'current_step': self.current_step,
            'max_steps': self.max_episode_steps,
            'cumulative_reward': round(self.cumulative_reward, 2),
            'queue_health': round(self.bug_queue.queue_health_score * 100, 1),
            'team_availability': round(self.team_state.availability_ratio * 100, 1),
            'critical_bugs': self.bug_queue.critical_count,
            'total_bugs': len(self.bug_queue.bugs),
            'open_bugs': len(open_bugs),
            'bugs': bugs_data,
            'developers': devs_data,
            'episode_metrics': {
                'total_processed': self.episode_metrics.total_bugs_processed,
                'resolved_within_sla': self.episode_metrics.bugs_resolved_within_sla,
                'escalation_rate': round(self.episode_metrics.escalation_rate, 2)
            },
            'action_history': self.action_history[-10:]  # Last 10 actions
        }

    def render(self):
        if self.render_mode == "human":
            print(f"\n=== Bug Triage (Step {self.current_step}/{self.max_episode_steps}) ===")
            print(f"Task: {self.task_level} | Reward: {self.cumulative_reward:.2f}")
            print(f"Queue Health: {self.bug_queue.queue_health_score:.2f}")
            print(f"Team Availability: {self.team_state.availability_ratio:.2f}")
            open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
            print(f"Open Bugs: {len(open_bugs)}")
            print("=" * 40)

    def close(self):
        pass