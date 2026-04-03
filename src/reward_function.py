"""
Reward function for Bug Triage RL Environment
"""

from typing import Dict
from .models import Bug, BugSeverity, BugStatus, ActionType, Reward, TeamState, BugQueue


class RewardCalculator:

    def __init__(self, task_level: str = "medium"):
        self.task_level = task_level
        self._setup_reward_parameters()

    def _setup_reward_parameters(self):
        if self.task_level == "easy":
            self.base_rewards = {
                ActionType.ASSIGN: 0.2,
                ActionType.ESCALATE: 0.0,
                ActionType.DEFER: -0.1,
                ActionType.CLOSE: 0.3
            }
            self.critical_bonus = 0.4
            self.sla_bonus = 0.2
            self.escalation_penalty = 0.1

        elif self.task_level == "medium":
            self.base_rewards = {
                ActionType.ASSIGN: 0.3,
                ActionType.ESCALATE: 0.2,
                ActionType.DEFER: -0.2,
                ActionType.CLOSE: 0.4
            }
            self.critical_bonus = 0.6
            self.sla_bonus = 0.4
            self.escalation_penalty = 0.3

        else:  # hard
            self.base_rewards = {
                ActionType.ASSIGN: 0.4,
                ActionType.ESCALATE: 0.3,
                ActionType.DEFER: -0.4,
                ActionType.CLOSE: 0.5
            }
            self.critical_bonus = 0.8
            self.sla_bonus = 0.6
            self.escalation_penalty = 0.5

    def calculate_reward(
        self,
        action: ActionType,
        bug: Bug,
        team_state: TeamState,
        bug_queue: BugQueue,
        previous_queue_health: float
    ) -> Reward:
        components = {}
        total = 0.0

        # Base reward
        base = self.base_rewards.get(action, 0.0)
        components["base_reward"] = base
        total += base

        # Severity modifier
        severity_mod = self._severity_modifier(bug, action)
        components["severity_modifier"] = severity_mod
        total += severity_mod

        # SLA reward
        sla_r = self._sla_reward(bug, action)
        components["sla_reward"] = sla_r
        total += sla_r

        # Efficiency reward
        eff_r = self._efficiency_reward(bug, team_state, action)
        components["efficiency_reward"] = eff_r
        total += eff_r

        # Queue health improvement
        health_r = (bug_queue.queue_health_score - previous_queue_health) * 0.5
        components["health_improvement"] = health_r
        total += health_r

        # Penalties
        penalty = self._penalties(bug, action, team_state)
        components["penalties"] = penalty
        total += penalty

        total = max(-1.0, min(1.0, total))

        return Reward(
            immediate_reward=total,
            cumulative_reward=0.0,
            reward_components=components,
            reason=self._explain(action, bug, components),
            efficiency_bonus=eff_r,
            penalty_applied=abs(penalty) if penalty < 0 else 0.0
        )

    def _severity_modifier(self, bug: Bug, action: ActionType) -> float:
        multipliers = {
            BugSeverity.CRITICAL: 2.0,
            BugSeverity.HIGH: 1.5,
            BugSeverity.MEDIUM: 1.0,
            BugSeverity.LOW: 0.5
        }
        m = multipliers[bug.severity]
        if bug.severity == BugSeverity.CRITICAL:
            if action in [ActionType.ASSIGN, ActionType.ESCALATE]:
                return self.critical_bonus * m
            elif action == ActionType.DEFER:
                return -self.critical_bonus
        elif bug.severity == BugSeverity.HIGH:
            if action == ActionType.ASSIGN:
                return self.critical_bonus * 0.5
            elif action == ActionType.DEFER:
                return -self.critical_bonus * 0.5
        return 0.0

    def _sla_reward(self, bug: Bug, action: ActionType) -> float:
        if bug.age_hours >= bug.sla_hours:
            # Past SLA
            if action == ActionType.ESCALATE:
                return self.sla_bonus
            elif action == ActionType.ASSIGN:
                return self.sla_bonus * 0.5
            elif action == ActionType.DEFER:
                return -self.sla_bonus
            return 0.0

        # Within SLA — reward urgency-aware decisions
        time_ratio = (bug.sla_hours - bug.age_hours) / bug.sla_hours
        if action == ActionType.ASSIGN and time_ratio < 0.3:
            return self.sla_bonus * (1 - time_ratio)
        return 0.0

    def _efficiency_reward(self, bug: Bug, team_state: TeamState, action: ActionType) -> float:
        if action != ActionType.ASSIGN:
            return 0.0
        available = [d for d in team_state.developers if d.is_available]
        if not available:
            return -0.3
        loads = [d.current_load for d in available]
        if max(loads) > 0:
            balance = 1.0 - (max(loads) - min(loads)) / max(loads)
            return 0.2 * balance
        return 0.1

    def _penalties(self, bug: Bug, action: ActionType, team_state: TeamState) -> float:
        p = 0.0
        # Unnecessary escalation
        if action == ActionType.ESCALATE and bug.severity in [BugSeverity.LOW, BugSeverity.MEDIUM]:
            p -= self.escalation_penalty
        # Deferring high priority
        if action == ActionType.DEFER and bug.severity in [BugSeverity.HIGH, BugSeverity.CRITICAL]:
            p -= self.escalation_penalty * 2
        # Closing too quickly
        if action == ActionType.CLOSE and self.task_level != "easy" and bug.age_hours < 1:
            p -= 0.2
        # Overloading team
        if action == ActionType.ASSIGN and team_state.availability_ratio < 0.1:
            p -= 0.2
        return p

    def _explain(self, action: ActionType, bug: Bug, components: Dict) -> str:
        parts = []
        if components.get("base_reward", 0):
            parts.append(f"Base({action.value}): {components['base_reward']:.2f}")
        if components.get("severity_modifier", 0):
            parts.append(f"Severity({bug.severity.value}): {components['severity_modifier']:.2f}")
        if components.get("sla_reward", 0):
            parts.append(f"SLA: {components['sla_reward']:.2f}")
        if components.get("penalties", 0) < 0:
            parts.append(f"Penalty: {components['penalties']:.2f}")
        return " | ".join(parts) if parts else "No significant factors"

    def calculate_episode_success_score(self, final_metrics, initial_health, final_health) -> float:
        sla_score = final_metrics.get("bugs_resolved_within_sla", 0) / max(final_metrics.get("total_bugs_processed", 1), 1)
        health_score = max(0.0, min(1.0, 0.5 + (final_health - initial_health)))
        utilization_score = min(1.0, final_metrics.get("team_utilization", 0) / 0.8)
        efficiency_score = max(0.0, 1.0 - final_metrics.get("avg_resolution_time", 0) / 100)
        return max(0.0, min(1.0,
            sla_score * 0.4 +
            health_score * 0.3 +
            utilization_score * 0.2 +
            efficiency_score * 0.1
        ))