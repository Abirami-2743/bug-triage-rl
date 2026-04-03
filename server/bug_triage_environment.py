"""
OpenEnv-compliant Bug Triage Environment Server
Implements step() / reset() / state() API
"""

import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models import (
    Bug, BugQueue, TeamState, Developer, DeveloperSkill,
    Action, ActionType, BugStatus, BugSeverity, EpisodeMetrics
)
from src.bug_generator import BugGenerator
from src.reward_function import RewardCalculator


class BugTriageEnvironment:
    """
    OpenEnv-compliant Bug Triage Environment
    Implements: reset() / step() / state()
    """

    def __init__(self, task_level: str = "medium", max_steps: int = 100):
        self.task_level = task_level
        self.max_steps = max_steps
        self.bug_generator = BugGenerator(task_level)
        self.reward_calculator = RewardCalculator(task_level)

        self.current_step = 0
        self.cumulative_reward = 0.0
        self.bug_queue = BugQueue(bugs=[])
        self.team_state = TeamState(developers=[])
        self.episode_metrics = EpisodeMetrics()
        self.episode_id = 0
        self.done = False

        self._setup_team()

    def _setup_team(self):
        if self.task_level == "easy":
            configs = [
                ("dev-001", "Alice", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Bob", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol", [DeveloperSkill.FULLSTACK])
            ]
        elif self.task_level == "medium":
            configs = [
                ("dev-001", "Sarah Chen", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Marcus Johnson", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol White", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David Kim", [DeveloperSkill.FULLSTACK])
            ]
        else:
            configs = [
                ("dev-001", "Sarah Chen", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Marcus Johnson", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol White", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David Kim", [DeveloperSkill.FULLSTACK]),
                ("dev-005", "Eve Torres", [DeveloperSkill.SECURITY, DeveloperSkill.BACKEND]),
                ("dev-006", "Frank Lee", [DeveloperSkill.DEVOPS, DeveloperSkill.DATABASE]),
                ("dev-007", "Grace Park", [DeveloperSkill.FRONTEND, DeveloperSkill.FULLSTACK]),
                ("dev-008", "Henry Adams", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE])
            ]

        self.team_state = TeamState(developers=[
            Developer(id=id_, name=name, skills=skills, max_capacity=3)
            for id_, name, skills in configs
        ])
        self.team_state.update_metrics()

    def reset(self) -> dict:
        """Reset environment — returns initial observation"""
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.episode_id += 1
        self.episode_metrics = EpisodeMetrics()

        bugs = self.bug_generator.generate_bug_batch()
        self.bug_queue = BugQueue(bugs=bugs)
        self.bug_queue.update_metrics()

        for dev in self.team_state.developers:
            dev.current_load = 0
            dev.availability = True
        self.team_state.update_metrics()

        return self._get_observation()

    def step(self, action: dict) -> dict:
        """
        Execute action — returns {observation, reward, done, info}
        action format: {action_type, bug_id, developer_id}
        """
        if self.done:
            return {
                "observation": self._get_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"error": "Episode already done"}
            }

        # Parse action
        action_type_str = action.get("action_type", "defer")
        bug_id = action.get("bug_id", "")
        developer_id = action.get("developer_id", None)

        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.DEFER

        # Find bug
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        bug = next((b for b in open_bugs if b.id == bug_id), None)

        if not bug and open_bugs:
            bug = open_bugs[0]

        if not bug:
            self.current_step += 1
            self.done = self.current_step >= self.max_steps
            return {
                "observation": self._get_observation(),
                "reward": -0.1,
                "done": self.done,
                "info": {"error": "No open bugs found"}
            }

        # Calculate reward
        prev_health = self.bug_queue.queue_health_score
        reward_obj = self.reward_calculator.calculate_reward(
            action=action_type,
            bug=bug,
            team_state=self.team_state,
            bug_queue=self.bug_queue,
            previous_queue_health=prev_health
        )

        # Apply action
        self._apply_action(action_type, bug, developer_id)

        self.cumulative_reward += reward_obj.immediate_reward
        self.current_step += 1

        # Stream new bugs in hard mode
        if self.task_level == "hard" and self.current_step % 10 == 0:
            new_bug = self.bug_generator.generate_streaming_bug(self.current_step)
            self.bug_queue.bugs.append(new_bug)
            self.bug_queue.update_metrics()

        open_remaining = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        terminated = len(open_remaining) == 0
        truncated = self.current_step >= self.max_steps
        self.done = terminated or truncated

        return {
            "observation": self._get_observation(),
            "reward": round(reward_obj.immediate_reward, 4),
            "done": self.done,
            "info": {
                "action_type": action_type.value,
                "bug_id": bug.id,
                "bug_severity": bug.severity.value,
                "reward_reason": reward_obj.reason,
                "reward_components": reward_obj.reward_components,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "step": self.current_step,
                "open_bugs": len(open_remaining)
            }
        }

    def state(self) -> dict:
        """Returns current episode state metadata"""
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        return {
            "episode_id": self.episode_id,
            "step_count": self.current_step,
            "max_steps": self.max_steps,
            "task_level": self.task_level,
            "done": self.done,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "queue_health": round(self.bug_queue.queue_health_score, 4),
            "team_availability": round(self.team_state.availability_ratio, 4),
            "open_bugs": len(open_bugs),
            "total_bugs": len(self.bug_queue.bugs),
            "critical_bugs": self.bug_queue.critical_count,
            "bugs_processed": self.episode_metrics.total_bugs_processed,
            "resolved_within_sla": self.episode_metrics.bugs_resolved_within_sla
        }

    def grade(self) -> float:
        """
        Returns episode score 0.0–1.0 for grader
        Used by task graders to evaluate agent performance
        """
        return self.reward_calculator.calculate_episode_success_score(
            {
                "bugs_resolved_within_sla": self.episode_metrics.bugs_resolved_within_sla,
                "total_bugs_processed": self.episode_metrics.total_bugs_processed,
                "team_utilization": 1.0 - self.team_state.availability_ratio,
                "avg_resolution_time": self.episode_metrics.avg_resolution_time
            },
            initial_health=0.5,
            final_health=self.bug_queue.queue_health_score
        )

    def _apply_action(self, action_type: ActionType, bug: Bug, developer_id: str = None):
        if action_type == ActionType.ASSIGN:
            available = [d for d in self.team_state.developers if d.is_available]
            if developer_id:
                dev = next((d for d in available if d.id == developer_id), None)
            else:
                dev = available[0] if available else None

            if dev:
                bug.status = BugStatus.IN_PROGRESS
                bug.assigned_to = dev.id
                dev.current_load += 1
                if dev.current_load >= dev.max_capacity:
                    dev.availability = False
                self.episode_metrics.total_bugs_processed += 1
                if bug.age_hours <= bug.sla_hours:
                    self.episode_metrics.bugs_resolved_within_sla += 1

        elif action_type == ActionType.ESCALATE:
            bug.status = BugStatus.ESCALATED
            bug.escalation_level += 1
            self.episode_metrics.total_bugs_processed += 1

        elif action_type == ActionType.DEFER:
            bug.status = BugStatus.DEFERRED
            bug.age_hours += 2

        elif action_type == ActionType.CLOSE:
            bug.status = BugStatus.CLOSED
            self.episode_metrics.total_bugs_processed += 1
            if bug.age_hours <= bug.sla_hours:
                self.episode_metrics.bugs_resolved_within_sla += 1

        self.bug_queue.update_metrics()
        self.team_state.update_metrics()

    def _get_observation(self) -> dict:
        open_bugs = [b for b in self.bug_queue.bugs if b.status == BugStatus.OPEN]
        open_bugs.sort(key=lambda b: b.priority_score, reverse=True)

        bugs_data = [{
            "id": b.id,
            "title": b.title,
            "severity": b.severity.value,
            "bug_type": b.bug_type.value,
            "status": b.status.value,
            "age_hours": b.age_hours,
            "affected_users": b.affected_users,
            "priority_score": round(b.priority_score, 3),
            "sla_hours": b.sla_hours,
            "sla_usage_pct": round(min(b.age_hours / b.sla_hours * 100, 200), 1),
            "is_overdue": b.age_hours > b.sla_hours
        } for b in open_bugs[:20]]

        devs_data = [{
            "id": d.id,
            "name": d.name,
            "skills": [s.value for s in d.skills],
            "current_load": d.current_load,
            "max_capacity": d.max_capacity,
            "is_available": d.is_available,
            "utilization_pct": round(d.current_load / d.max_capacity * 100, 1)
        } for d in self.team_state.developers]

        return {
            "bug_queue": {
                "open_bugs": bugs_data,
                "total_count": len(self.bug_queue.bugs),
                "open_count": len(open_bugs),
                "critical_count": self.bug_queue.critical_count,
                "avg_age_hours": round(self.bug_queue.avg_age_hours, 2),
                "queue_health_score": round(self.bug_queue.queue_health_score, 4)
            },
            "team": {
                "developers": devs_data,
                "total_capacity": self.team_state.total_capacity,
                "current_load": self.team_state.current_load,
                "availability_ratio": round(self.team_state.availability_ratio, 4)
            },
            "metrics": {
                "step": self.current_step,
                "max_steps": self.max_steps,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "task_level": self.task_level
            }
        }