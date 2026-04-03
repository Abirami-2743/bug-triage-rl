"""
Task definitions and graders for the 3 difficulty levels
Implements automated grading with clear success criteria.
"""

from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod

from .environment import BugTriageEnv
from .models import EpisodeMetrics, BugSeverity


class TaskGrader(ABC):
    """Abstract base class for task graders"""
    
    def __init__(self, task_level: str):
        self.task_level = task_level
    
    @abstractmethod
    def grade_episode(self, env: BugTriageEnv, episode_metrics: EpisodeMetrics) -> float:
        """
        Grade an episode and return score (0.0 - 1.0)
        
        Args:
            env: Environment instance
            episode_metrics: Episode performance metrics
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for this task"""
        pass


class EasyTaskGrader(TaskGrader):
    """Grader for Easy task: Single Priority Queue"""
    
    def __init__(self):
        super().__init__("easy")
        
        # Success thresholds for easy task
        self.thresholds = {
            "min_bugs_processed": 8,  # At least 8 out of 10 bugs
            "min_sla_compliance": 0.7,  # 70% within SLA
            "max_avg_resolution_time": 24,  # Max 24 hours average
            "min_queue_health_improvement": 0.2  # Improve queue health by 20%
        }
    
    def grade_episode(self, env: BugTriageEnv, episode_metrics: EpisodeMetrics) -> float:
        """Grade easy task performance"""
        score_components = {}
        total_score = 0.0
        
        # Component 1: Bug processing (40%)
        bugs_processed = episode_metrics.total_bugs_processed
        processing_score = min(1.0, bugs_processed / self.thresholds["min_bugs_processed"])
        score_components["bug_processing"] = processing_score
        total_score += processing_score * 0.4
        
        # Component 2: SLA compliance (30%)
        if bugs_processed > 0:
            sla_compliance = episode_metrics.bugs_resolved_within_sla / bugs_processed
        else:
            sla_compliance = 0.0
        
        sla_score = min(1.0, sla_compliance / self.thresholds["min_sla_compliance"])
        score_components["sla_compliance"] = sla_score
        total_score += sla_score * 0.3
        
        # Component 3: Resolution time (20%)
        avg_time = episode_metrics.avg_resolution_time
        if avg_time > 0:
            time_score = max(0.0, 1.0 - (avg_time / self.thresholds["max_avg_resolution_time"]))
        else:
            time_score = 1.0
        score_components["resolution_time"] = time_score
        total_score += time_score * 0.2
        
        # Component 4: Queue health (10%)
        health_improvement = env.bug_queue.queue_health_score - env.initial_queue_health
        health_score = max(0.0, health_improvement / self.thresholds["min_queue_health_improvement"])
        score_components["queue_health"] = health_score
        total_score += health_score * 0.1
        
        return max(0.0, min(1.0, total_score))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for easy task"""
        return {
            "description": "Process at least 8 out of 10 bugs efficiently",
            "requirements": [
                f"Process ≥ {self.thresholds['min_bugs_processed']} bugs",
                f"≥ {self.thresholds['min_sla_compliance']*100:.0f}% SLA compliance",
                f"Average resolution time ≤ {self.thresholds['max_avg_resolution_time']} hours",
                f"Improve queue health by ≥ {self.thresholds['min_queue_health_improvement']*100:.0f}%"
            ],
            "passing_score": 0.7,  # 70% to pass
            "weighting": {
                "bug_processing": 0.4,
                "sla_compliance": 0.3,
                "resolution_time": 0.2,
                "queue_health": 0.1
            }
        }


class MediumTaskGrader(TaskGrader):
    """Grader for Medium task: Multi-Queue Management"""
    
    def __init__(self):
        super().__init__("medium")
        
        # Success thresholds for medium task
        self.thresholds = {
            "min_bugs_processed": 20,  # At least 20 out of 25 bugs
            "min_sla_compliance": 0.8,  # 80% within SLA
            "max_avg_resolution_time": 18,  # Max 18 hours average
            "min_team_utilization": 0.6,  # At least 60% team utilization
            "max_escalation_rate": 0.2  # Max 20% escalation rate
        }
    
    def grade_episode(self, env: BugTriageEnv, episode_metrics: EpisodeMetrics) -> float:
        """Grade medium task performance"""
        score_components = {}
        total_score = 0.0
        
        # Component 1: Bug processing (30%)
        bugs_processed = episode_metrics.total_bugs_processed
        processing_score = min(1.0, bugs_processed / self.thresholds["min_bugs_processed"])
        score_components["bug_processing"] = processing_score
        total_score += processing_score * 0.3
        
        # Component 2: SLA compliance (25%)
        if bugs_processed > 0:
            sla_compliance = episode_metrics.bugs_resolved_within_sla / bugs_processed
        else:
            sla_compliance = 0.0
        
        sla_score = min(1.0, sla_compliance / self.thresholds["min_sla_compliance"])
        score_components["sla_compliance"] = sla_score
        total_score += sla_score * 0.25
        
        # Component 3: Team utilization (25%)
        utilization_score = min(1.0, episode_metrics.team_utilization / self.thresholds["min_team_utilization"])
        score_components["team_utilization"] = utilization_score
        total_score += utilization_score * 0.25
        
        # Component 4: Resolution time (10%)
        avg_time = episode_metrics.avg_resolution_time
        if avg_time > 0:
            time_score = max(0.0, 1.0 - (avg_time / self.thresholds["max_avg_resolution_time"]))
        else:
            time_score = 1.0
        score_components["resolution_time"] = time_score
        total_score += time_score * 0.1
        
        # Component 5: Escalation management (10%)
        escalation_penalty = max(0.0, episode_metrics.escalation_rate - self.thresholds["max_escalation_rate"])
        escalation_score = max(0.0, 1.0 - (escalation_penalty * 5))  # Heavy penalty for over-escalation
        score_components["escalation_management"] = escalation_score
        total_score += escalation_score * 0.1
        
        return max(0.0, min(1.0, total_score))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for medium task"""
        return {
            "description": "Manage multi-queue with resource constraints efficiently",
            "requirements": [
                f"Process ≥ {self.thresholds['min_bugs_processed']} out of 25 bugs",
                f"≥ {self.thresholds['min_sla_compliance']*100:.0f}% SLA compliance",
                f"Average resolution time ≤ {self.thresholds['max_avg_resolution_time']} hours",
                f"Maintain ≥ {self.thresholds['min_team_utilization']*100:.0f}% team utilization",
                f"Keep escalation rate ≤ {self.thresholds['max_escalation_rate']*100:.0f}%"
            ],
            "passing_score": 0.75,  # 75% to pass
            "weighting": {
                "bug_processing": 0.3,
                "sla_compliance": 0.25,
                "team_utilization": 0.25,
                "resolution_time": 0.1,
                "escalation_management": 0.1
            }
        }


class HardTaskGrader(TaskGrader):
    """Grader for Hard task: Enterprise-Scale Triage"""
    
    def __init__(self):
        super().__init__("hard")
        
        # Success thresholds for hard task
        self.thresholds = {
            "min_bugs_processed": 40,  # At least 40 out of 50+ bugs
            "min_sla_compliance": 0.85,  # 85% within SLA
            "max_avg_resolution_time": 12,  # Max 12 hours average
            "min_team_utilization": 0.7,  # At least 70% team utilization
            "max_critical_overdue": 2,  # Max 2 critical bugs overdue
            "min_queue_health_final": 0.6  # Final queue health ≥ 60%
        }
    
    def grade_episode(self, env: BugTriageEnv, episode_metrics: EpisodeMetrics) -> float:
        """Grade hard task performance"""
        score_components = {}
        total_score = 0.0
        
        # Component 1: Bug processing (25%)
        bugs_processed = episode_metrics.total_bugs_processed
        processing_score = min(1.0, bugs_processed / self.thresholds["min_bugs_processed"])
        score_components["bug_processing"] = processing_score
        total_score += processing_score * 0.25
        
        # Component 2: SLA compliance (25%)
        if bugs_processed > 0:
            sla_compliance = episode_metrics.bugs_resolved_within_sla / bugs_processed
        else:
            sla_compliance = 0.0
        
        sla_score = min(1.0, sla_compliance / self.thresholds["min_sla_compliance"])
        score_components["sla_compliance"] = sla_score
        total_score += sla_score * 0.25
        
        # Component 3: Critical bug management (20%)
        critical_overdue = sum(
            1 for bug in env.bug_queue.bugs 
            if bug.severity == BugSeverity.CRITICAL and bug.age_hours > bug.sla_hours
        )
        critical_score = max(0.0, 1.0 - (critical_overdue / self.thresholds["max_critical_overdue"]))
        score_components["critical_management"] = critical_score
        total_score += critical_score * 0.2
        
        # Component 4: Team utilization (15%)
        utilization_score = min(1.0, episode_metrics.team_utilization / self.thresholds["min_team_utilization"])
        score_components["team_utilization"] = utilization_score
        total_score += utilization_score * 0.15
        
        # Component 5: Resolution time (10%)
        avg_time = episode_metrics.avg_resolution_time
        if avg_time > 0:
            time_score = max(0.0, 1.0 - (avg_time / self.thresholds["max_avg_resolution_time"]))
        else:
            time_score = 1.0
        score_components["resolution_time"] = time_score
        total_score += time_score * 0.1
        
        # Component 6: Queue health (5%)
        final_health = env.bug_queue.queue_health_score
        health_score = min(1.0, final_health / self.thresholds["min_queue_health_final"])
        score_components["queue_health"] = health_score
        total_score += health_score * 0.05
        
        return max(0.0, min(1.0, total_score))
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get success criteria for hard task"""
        return {
            "description": "Enterprise-scale triage with SLA constraints and streaming bugs",
            "requirements": [
                f"Process ≥ {self.thresholds['min_bugs_processed']} bugs (with streaming arrivals)",
                f"≥ {self.thresholds['min_sla_compliance']*100:.0f}% SLA compliance",
                f"Average resolution time ≤ {self.thresholds['max_avg_resolution_time']} hours",
                f"Maintain ≥ {self.thresholds['min_team_utilization']*100:.0f}% team utilization",
                f"Keep ≤ {self.thresholds['max_critical_overdue']} critical bugs overdue",
                f"Final queue health ≥ {self.thresholds['min_queue_health_final']*100:.0f}%"
            ],
            "passing_score": 0.8,  # 80% to pass
            "weighting": {
                "bug_processing": 0.25,
                "sla_compliance": 0.25,
                "critical_management": 0.2,
                "team_utilization": 0.15,
                "resolution_time": 0.1,
                "queue_health": 0.05
            }
        }


class TaskManager:
    """Manages task definitions and grading"""
    
    def __init__(self):
        self.graders = {
            "easy": EasyTaskGrader(),
            "medium": MediumTaskGrader(),
            "hard": HardTaskGrader()
        }
    
    def grade_episode(self, task_level: str, env: BugTriageEnv, episode_metrics: EpisodeMetrics) -> float:
        """Grade an episode for the specified task level"""
        if task_level not in self.graders:
            raise ValueError(f"Unknown task level: {task_level}")
        
        return self.graders[task_level].grade_episode(env, episode_metrics)
    
    def get_task_info(self, task_level: str) -> Dict[str, Any]:
        """Get task information and success criteria"""
        if task_level not in self.graders:
            raise ValueError(f"Unknown task level: {task_level}")
        
        grader = self.graders[task_level]
        return {
            "task_level": task_level,
            "success_criteria": grader.get_success_criteria(),
            "description": grader.get_success_criteria()["description"]
        }
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all tasks"""
        return {
            level: self.get_task_info(level)
            for level in ["easy", "medium", "hard"]
        }
    
    def evaluate_baseline_performance(self, task_level: str, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate baseline random agent performance
        
        Args:
            task_level: Task difficulty
            num_episodes: Number of evaluation episodes
            
        Returns:
            Performance statistics
        """
        from .environment import BugTriageEnv
        import random
        
        scores = []
        
        for episode in range(num_episodes):
            env = BugTriageEnv(task_level=task_level)
            obs = env.reset()
            done = False
            
            while not done:
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
                if action_type == "assign":
                    available_devs = [dev for dev in env.team_state.developers if dev.is_available]
                    if available_devs:
                        developer_id = random.choice(available_devs).id
                    else:
                        # No available developers, skip assign action
                        continue
                
                # Create action
                from .models import Action, ActionType
                action = Action(
                    action_type=ActionType(action_type),
                    bug_id=bug_id,
                    developer_id=developer_id,
                    reasoning="Random agent action"
                )
                
                # Step environment
                obs, reward, done, info = env.step(action)
            
            # Grade episode
            score = self.grade_episode(task_level, env, env.episode_metrics)
            scores.append(score)
        
        return {
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
        }
