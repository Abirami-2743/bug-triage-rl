"""
Main Bug Triage RL Environment
Implements OpenEnv specification with step(), reset(), and state() methods.
"""

import random
from typing import Dict, List, Optional, Tuple, Any
import uuid

from .models import (
    Bug, BugQueue, TeamState, Developer, DeveloperSkill,
    Observation, Action, Reward, ActionType, BugStatus, BugSeverity,
    EnvironmentState, EpisodeMetrics
)
from .bug_generator import BugGenerator
from .reward_function import RewardCalculator


class BugTriageEnv:
    """
    Bug Triage & Escalation Desk RL Environment
    
    An OpenEnv-compliant environment where AI agents learn to triage and escalate
    software bugs intelligently.
    """
    
    def __init__(self, task_level: str = "medium", max_episode_steps: int = 100):
        """
        Initialize the environment
        
        Args:
            task_level: "easy", "medium", or "hard"
            max_episode_steps: Maximum steps per episode
        """
        self.task_level = task_level
        self.max_episode_steps = max_episode_steps
        
        # Initialize components
        self.bug_generator = BugGenerator(task_level)
        self.reward_calculator = RewardCalculator(task_level)
        
        # Environment state
        self.current_step = 0
        self.bug_queue: BugQueue = BugQueue(bugs=[])
        self.team_state: TeamState = TeamState(developers=[])
        self.cumulative_reward = 0.0
        self.episode_metrics: EpisodeMetrics = EpisodeMetrics()
        
        # Episode tracking
        self.initial_queue_health = 0.0
        self.action_history: List[Dict] = []
        
        # Setup team based on task level
        self._setup_team()
        
        # Initialize environment
        self.reset()
    
    def _setup_team(self):
        """Setup development team based on task difficulty"""
        if self.task_level == "easy":
            team_config = [
                ("dev-001", "Alice", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Bob", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE])
            ]
        elif self.task_level == "medium":
            team_config = [
                ("dev-001", "Alice", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Bob", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David", [DeveloperSkill.FULLSTACK])
            ]
        else:  # hard
            team_config = [
                ("dev-001", "Alice", [DeveloperSkill.FRONTEND, DeveloperSkill.BACKEND]),
                ("dev-002", "Bob", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE]),
                ("dev-003", "Carol", [DeveloperSkill.SECURITY, DeveloperSkill.DEVOPS]),
                ("dev-004", "David", [DeveloperSkill.FULLSTACK]),
                ("dev-005", "Eve", [DeveloperSkill.SECURITY, DeveloperSkill.BACKEND]),
                ("dev-006", "Frank", [DeveloperSkill.DEVOPS, DeveloperSkill.DATABASE]),
                ("dev-007", "Grace", [DeveloperSkill.FRONTEND, DeveloperSkill.FULLSTACK]),
                ("dev-008", "Henry", [DeveloperSkill.BACKEND, DeveloperSkill.DATABASE])
            ]
        
        developers = []
        for dev_id, name, skills in team_config:
            developer = Developer(
                id=dev_id,
                name=name,
                skills=skills,
                max_capacity=3 if self.task_level == "hard" else 2
            )
            developers.append(developer)
        
        self.team_state = TeamState(developers=developers)
        self.team_state.update_metrics()
    
    def reset(self) -> Observation:
        """
        Reset the environment for a new episode
        
        Returns:
            Initial observation
        """
        # Reset counters
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.action_history = []
        
        # Generate initial bug batch
        initial_bugs = self.bug_generator.generate_bug_batch()
        self.bug_queue = BugQueue(bugs=initial_bugs)
        self.bug_queue.update_metrics()
        
        # Reset team state
        for dev in self.team_state.developers:
            dev.current_load = 0
            dev.availability = True
        self.team_state.update_metrics()
        
        # Reset episode metrics
        self.episode_metrics = EpisodeMetrics()
        
        # Store initial queue health
        self.initial_queue_health = self.bug_queue.queue_health_score
        
        return self._create_observation()
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Validate action
        if not self._validate_action(action):
            # Invalid action - apply penalty
            reward = Reward(
                immediate_reward=-1.0,
                cumulative_reward=self.cumulative_reward - 1.0,
                reward_components={"invalid_action": -1.0},
                reason="Invalid action specified"
            )
            self.cumulative_reward += reward.immediate_reward
            obs = self._create_observation()
            return obs, reward.immediate_reward, False, {"error": "Invalid action"}
        
        # Find target bug
        target_bug = self._find_bug(action.bug_id)
        if not target_bug:
            # Bug not found - apply penalty
            reward = Reward(
                immediate_reward=-0.5,
                cumulative_reward=self.cumulative_reward - 0.5,
                reward_components={"bug_not_found": -0.5},
                reason=f"Bug {action.bug_id} not found"
            )
            self.cumulative_reward += reward.immediate_reward
            obs = self._create_observation()
            return obs, reward.immediate_reward, False, {"error": "Bug not found"}
        
        # Store previous queue health for reward calculation
        previous_health = self.bug_queue.queue_health_score
        
        # Execute action
        action_result = self._execute_action(action, target_bug)
        
        # Update environment state
        self._update_environment_state()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            action.action_type,
            target_bug,
            self.team_state,
            self.bug_queue,
            previous_health
        )
        reward.cumulative_reward = self.cumulative_reward + reward.immediate_reward
        self.cumulative_reward = reward.cumulative_reward
        
        # Record action
        self.action_history.append({
            "step": self.current_step,
            "action": action.dict(),
            "reward": reward.dict(),
            "bug_before": target_bug.dict(),
            "queue_health_before": previous_health,
            "queue_health_after": self.bug_queue.queue_health_score
        })
        
        # Increment step
        self.current_step += 1
        
        # Check episode termination
        done = self._check_episode_done()
        
        # Create observation
        obs = self._create_observation()
        
        # Additional info
        info = {
            "action_result": action_result,
            "episode_metrics": self.episode_metrics.dict(),
            "queue_health": self.bug_queue.queue_health_score,
            "team_utilization": self.team_state.availability_ratio
        }
        
        return obs, reward.immediate_reward, done, info
    
    def state(self) -> EnvironmentState:
        """
        Get current environment state (OpenEnv compliance)
        
        Returns:
            Complete environment state
        """
        return EnvironmentState(
            observation=self._create_observation(),
            action=self.action_history[-1]["action"] if self.action_history else None,
            reward=Reward(
                immediate_reward=self.action_history[-1]["reward"]["immediate_reward"] if self.action_history else 0.0,
                cumulative_reward=self.cumulative_reward,
                reward_components=self.action_history[-1]["reward"]["reward_components"] if self.action_history else {},
                reason=self.action_history[-1]["reward"]["reason"] if self.action_history else "No actions taken"
            ),
            done=self._check_episode_done(),
            info={
                "task_level": self.task_level,
                "current_step": self.current_step,
                "action_history_length": len(self.action_history)
            },
            episode_metrics=self.episode_metrics
        )
    
    def _validate_action(self, action: Action) -> bool:
        """Validate action parameters"""
        if not isinstance(action, Action):
            return False
        
        if action.action_type == ActionType.ASSIGN and not action.developer_id:
            return False
        
        # Check if developer exists for assignment
        if action.action_type == ActionType.ASSIGN:
            dev_exists = any(dev.id == action.developer_id for dev in self.team_state.developers)
            if not dev_exists:
                return False
        
        return True
    
    def _find_bug(self, bug_id: str) -> Optional[Bug]:
        """Find bug by ID"""
        for bug in self.bug_queue.bugs:
            if bug.id == bug_id:
                return bug
        return None
    
    def _execute_action(self, action: Action, bug: Bug) -> Dict[str, Any]:
        """Execute the specified action on the bug"""
        result = {"success": False, "message": "", "bug_changes": {}}
        
        if action.action_type == ActionType.ASSIGN:
            result = self._assign_bug(bug, action.developer_id)
        
        elif action.action_type == ActionType.ESCALATE:
            result = self._escalate_bug(bug)
        
        elif action.action_type == ActionType.DEFER:
            result = self._defer_bug(bug)
        
        elif action.action_type == ActionType.CLOSE:
            result = self._close_bug(bug)
        
        # Update episode metrics
        if result["success"]:
            self.episode_metrics.total_bugs_processed += 1
            
            if action.action_type == ActionType.CLOSE:
                if bug.time_to_resolution and bug.time_to_resolution <= bug.sla_hours:
                    self.episode_metrics.bugs_resolved_within_sla += 1
        
        return result
    
    def _assign_bug(self, bug: Bug, developer_id: str) -> Dict[str, Any]:
        """Assign bug to developer"""
        developer = next((dev for dev in self.team_state.developers if dev.id == developer_id), None)
        
        if not developer:
            return {"success": False, "message": "Developer not found"}
        
        if not developer.is_available:
            return {"success": False, "message": "Developer not available"}
        
        # Assign bug
        bug.assigned_to = developer_id
        bug.status = BugStatus.IN_PROGRESS
        developer.current_load += 1
        
        self.team_state.update_metrics()
        
        return {
            "success": True,
            "message": f"Bug {bug.id} assigned to {developer.name}",
            "bug_changes": {"assigned_to": developer_id, "status": BugStatus.IN_PROGRESS}
        }
    
    def _escalate_bug(self, bug: Bug) -> Dict[str, Any]:
        """Escalate bug to higher level"""
        bug.escalation_level += 1
        bug.status = BugStatus.ESCALATED
        
        self.episode_metrics.escalation_rate += 1 / max(self.episode_metrics.total_bugs_processed, 1)
        
        return {
            "success": True,
            "message": f"Bug {bug.id} escalated to level {bug.escalation_level}",
            "bug_changes": {"escalation_level": bug.escalation_level, "status": BugStatus.ESCALATED}
        }
    
    def _defer_bug(self, bug: Bug) -> Dict[str, Any]:
        """Defer bug for later"""
        bug.status = BugStatus.DEFERRED
        
        return {
            "success": True,
            "message": f"Bug {bug.id} deferred",
            "bug_changes": {"status": BugStatus.DEFERRED}
        }
    
    def _close_bug(self, bug: Bug) -> Dict[str, Any]:
        """Close bug as resolved"""
        bug.status = BugStatus.CLOSED
        bug.time_to_resolution = bug.age_hours
        
        # Update developer load if assigned
        if bug.assigned_to:
            developer = next((dev for dev in self.team_state.developers if dev.id == bug.assigned_to), None)
            if developer:
                developer.current_load = max(0, developer.current_load - 1)
                self.team_state.update_metrics()
        
        return {
            "success": True,
            "message": f"Bug {bug.id} closed after {bug.age_hours} hours",
            "bug_changes": {"status": BugStatus.CLOSED, "time_to_resolution": bug.age_hours}
        }
    
    def _update_environment_state(self):
        """Update environment state after actions"""
        # Age bugs
        for bug in self.bug_queue.bugs:
            bug.age_hours += 1
        
        # Remove closed bugs from queue
        self.bug_queue.bugs = [bug for bug in self.bug_queue.bugs if bug.status != BugStatus.CLOSED]
        
        # Add new bugs in hard mode (streaming)
        if self.task_level == "hard" and random.random() < 0.3:  # 30% chance of new bug
            new_bug = self.bug_generator.generate_streaming_bug(self.current_step)
            self.bug_queue.bugs.append(new_bug)
        
        # Update queue metrics
        self.bug_queue.update_metrics()
        
        # Update episode metrics
        if self.bug_queue.bugs:
            resolution_times = [bug.age_hours for bug in self.bug_queue.bugs if bug.time_to_resolution]
            if resolution_times:
                self.episode_metrics.avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        self.episode_metrics.team_utilization = 1.0 - self.team_state.availability_ratio
    
    def _check_episode_done(self) -> bool:
        """Check if episode should terminate"""
        # Episode ends if max steps reached
        if self.current_step >= self.max_episode_steps:
            return True
        
        # Episode ends if all bugs processed (in easy/medium mode)
        if self.task_level != "hard" and len(self.bug_queue.bugs) == 0:
            return True
        
        # Episode ends if too many critical bugs overdue (in hard mode)
        if self.task_level == "hard":
            overdue_critical = sum(
                1 for bug in self.bug_queue.bugs 
                if bug.severity == BugSeverity.CRITICAL and bug.age_hours > bug.sla_hours
            )
            if overdue_critical > 5:  # Too many critical bugs overdue
                return True
        
        return False
    
    def _create_observation(self) -> Observation:
        """Create current observation"""
        system_metrics = {
            "queue_health": self.bug_queue.queue_health_score,
            "team_availability": self.team_state.availability_ratio,
            "critical_bugs_count": self.bug_queue.critical_count,
            "avg_bug_age": self.bug_queue.avg_age_hours,
            "escalation_rate": self.episode_metrics.escalation_rate
        }
        
        return Observation(
            bug_queue=self.bug_queue,
            team_state=self.team_state,
            system_metrics=system_metrics,
            current_step=self.current_step,
            total_steps=self.max_episode_steps,
            time_remaining=max(0, self.max_episode_steps - self.current_step)
        )
    
    def get_action_space(self) -> List[ActionType]:
        """Get available actions"""
        return list(ActionType)
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specification"""
        return {
            "bug_queue": {
                "bugs": "List[Bug]",
                "total_count": "int",
                "critical_count": "int", 
                "avg_age_hours": "float",
                "queue_health_score": "float"
            },
            "team_state": {
                "developers": "List[Developer]",
                "total_capacity": "int",
                "current_load": "int",
                "availability_ratio": "float"
            },
            "system_metrics": "Dict[str, float]",
            "current_step": "int",
            "total_steps": "int",
            "time_remaining": "float"
        }
    
    def render(self, mode: str = "human"):
        """Render environment state (for debugging)"""
        if mode == "human":
            print(f"\n=== Bug Triage Environment (Step {self.current_step}) ===")
            print(f"Task Level: {self.task_level}")
            print(f"Queue Health: {self.bug_queue.queue_health_score:.2f}")
            print(f"Team Availability: {self.team_state.availability_ratio:.2f}")
            print(f"Total Bugs: {len(self.bug_queue.bugs)} (Critical: {self.bug_queue.critical_count})")
            
            print("\nTop 5 Bugs:")
            for i, bug in enumerate(self.bug_queue.bugs[:5]):
                print(f"  {i+1}. {bug.id} - {bug.severity.value} - {bug.title[:50]}...")
            
            print("\nTeam Status:")
            for dev in self.team_state.developers:
                status = f"{dev.current_load}/{dev.max_capacity}" if dev.is_available else "FULL"
                print(f"  {dev.name}: {status}")
            
            print(f"\nCumulative Reward: {self.cumulative_reward:.2f}")
            print("=" * 50)
