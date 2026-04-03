"""
Pydantic models for the Bug Triage RL Environment
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class BugSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BugType(str, Enum):
    UI = "ui"
    BACKEND = "backend"
    DATABASE = "database"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


class BugStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    CLOSED = "closed"


class ActionType(str, Enum):
    ASSIGN = "assign"
    ESCALATE = "escalate"
    DEFER = "defer"
    CLOSE = "close"


class DeveloperSkill(str, Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"
    FULLSTACK = "fullstack"
    SECURITY = "security"
    DEVOPS = "devops"
    DATABASE = "database"


class Bug(BaseModel):
    id: str
    title: str
    severity: BugSeverity
    bug_type: BugType
    status: BugStatus = BugStatus.OPEN
    age_hours: int = 0
    affected_users: int = 1
    priority_score: float = 0.5
    assigned_to: Optional[str] = None
    escalation_level: int = 0
    sla_hours: int = 72
    time_to_resolution: Optional[int] = None

    @validator('priority_score')
    def validate_priority_score(cls, v):
        return max(0.0, min(1.0, v))


class Developer(BaseModel):
    id: str
    name: str
    skills: List[DeveloperSkill]
    current_load: int = 0
    max_capacity: int = 3
    availability: bool = True

    @property
    def is_available(self) -> bool:
        return self.availability and self.current_load < self.max_capacity


class TeamState(BaseModel):
    developers: List[Developer]
    total_capacity: int = 0
    current_load: int = 0
    availability_ratio: float = 1.0

    def update_metrics(self):
        self.current_load = sum(dev.current_load for dev in self.developers)
        self.total_capacity = sum(dev.max_capacity for dev in self.developers)
        self.availability_ratio = (
            1.0 - (self.current_load / self.total_capacity)
            if self.total_capacity > 0 else 0.0
        )


class BugQueue(BaseModel):
    bugs: List[Bug]
    total_count: int = 0
    critical_count: int = 0
    avg_age_hours: float = 0.0
    queue_health_score: float = 1.0

    def update_metrics(self):
        self.total_count = len(self.bugs)
        self.critical_count = sum(
            1 for bug in self.bugs if bug.severity == BugSeverity.CRITICAL
        )
        if self.bugs:
            self.avg_age_hours = sum(bug.age_hours for bug in self.bugs) / len(self.bugs)
            age_penalty = min(self.avg_age_hours / 100.0, 1.0)
            critical_penalty = min(self.critical_count / 10.0, 1.0)
            self.queue_health_score = 1.0 - (age_penalty + critical_penalty) / 2.0
        else:
            self.avg_age_hours = 0.0
            self.queue_health_score = 1.0


class Action(BaseModel):
    action_type: ActionType
    bug_id: str
    developer_id: Optional[str] = None
    reasoning: Optional[str] = None


class Reward(BaseModel):
    immediate_reward: float
    cumulative_reward: float = 0.0
    reward_components: Dict[str, float] = {}
    reason: str = ""
    efficiency_bonus: float = 0.0
    penalty_applied: float = 0.0


class EpisodeMetrics(BaseModel):
    total_bugs_processed: int = 0
    bugs_resolved_within_sla: int = 0
    avg_resolution_time: float = 0.0
    team_utilization: float = 0.0
    escalation_rate: float = 0.0

    def calculate_success_score(self) -> float:
        sla_score = self.bugs_resolved_within_sla / max(self.total_bugs_processed, 1)
        utilization_score = min(self.team_utilization, 1.0)
        escalation_penalty = min(self.escalation_rate, 0.5)
        return max(0.0, (sla_score + utilization_score) / 2.0 - escalation_penalty)


class EnvironmentState(BaseModel):
    done: bool = False
    info: Dict[str, Any] = {}
    episode_metrics: EpisodeMetrics = EpisodeMetrics()

    class Config:
        arbitrary_types_allowed = True