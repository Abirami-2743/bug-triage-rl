"""
Bug Generator for creating realistic bug scenarios
"""

import random
from typing import List, Dict

from .models import Bug, BugSeverity, BugType, BugStatus


class BugGenerator:

    BUG_TEMPLATES = {
        BugType.UI: [
            "Button not responding to clicks",
            "Layout broken on mobile devices",
            "Modal dialog not closing properly",
            "Form validation not working",
            "Navigation menu items misaligned"
        ],
        BugType.BACKEND: [
            "API endpoint returning 500 errors",
            "Database connection timeout",
            "User authentication failing",
            "Data processing pipeline stuck",
            "Memory leak in background service"
        ],
        BugType.DATABASE: [
            "Slow query performance",
            "Deadlock in transaction processing",
            "Data corruption in user table",
            "Index not being used properly",
            "Migration script failing"
        ],
        BugType.SECURITY: [
            "SQL injection vulnerability",
            "Cross-site scripting (XSS) risk",
            "Authentication bypass possibility",
            "Sensitive data exposure",
            "Insecure API endpoints"
        ],
        BugType.PERFORMANCE: [
            "Page load time exceeding 5 seconds",
            "High CPU usage during peak hours",
            "Memory consumption growing indefinitely",
            "Database queries taking too long",
            "API response time degradation"
        ],
        BugType.INTEGRATION: [
            "Third-party API connection failing",
            "Webhook not delivering events",
            "Payment gateway integration issues",
            "Email service not sending notifications",
            "File upload service timeout"
        ]
    }

    SLA_REQUIREMENTS = {
        BugSeverity.CRITICAL: 4,
        BugSeverity.HIGH: 24,
        BugSeverity.MEDIUM: 72,
        BugSeverity.LOW: 168
    }

    def __init__(self, task_level: str = "medium"):
        self.task_level = task_level
        self._setup_generation_params()

    def _setup_generation_params(self):
        if self.task_level == "easy":
            self.max_bugs = 8
            self.severity_weights = {
                BugSeverity.LOW: 0.5,
                BugSeverity.MEDIUM: 0.4,
                BugSeverity.HIGH: 0.1,
                BugSeverity.CRITICAL: 0.0
            }
            self.max_affected_users = 50

        elif self.task_level == "medium":
            self.max_bugs = 15
            self.severity_weights = {
                BugSeverity.LOW: 0.3,
                BugSeverity.MEDIUM: 0.4,
                BugSeverity.HIGH: 0.2,
                BugSeverity.CRITICAL: 0.1
            }
            self.max_affected_users = 500

        else:  # hard
            self.max_bugs = 25
            self.severity_weights = {
                BugSeverity.LOW: 0.1,
                BugSeverity.MEDIUM: 0.2,
                BugSeverity.HIGH: 0.4,
                BugSeverity.CRITICAL: 0.3
            }
            self.max_affected_users = 5000

    def generate_bug_batch(self, count: int = None) -> List[Bug]:
        if count is None:
            count = self.max_bugs
        return [self._generate_single_bug(i) for i in range(count)]

    def _generate_single_bug(self, index: int) -> Bug:
        bug_types = list(BugType)
        bug_type = random.choice(bug_types)

        severities = list(BugSeverity)
        weights = [self.severity_weights[s] for s in severities]
        severity = self._weighted_choice(severities, weights)

        title = random.choice(self.BUG_TEMPLATES[bug_type])

        base_priority = {
            BugSeverity.LOW: 0.2,
            BugSeverity.MEDIUM: 0.5,
            BugSeverity.HIGH: 0.8,
            BugSeverity.CRITICAL: 1.0
        }[severity]

        priority_score = max(0.1, min(1.0, base_priority + random.uniform(-0.1, 0.1)))

        if severity == BugSeverity.CRITICAL:
            affected_users = random.randint(100, self.max_affected_users)
        elif severity == BugSeverity.HIGH:
            affected_users = random.randint(20, max(20, self.max_affected_users // 5))
        elif severity == BugSeverity.MEDIUM:
            affected_users = random.randint(5, max(5, self.max_affected_users // 20))
        else:
            affected_users = random.randint(1, max(1, self.max_affected_users // 50))

        if self.task_level == "hard":
            initial_age = random.randint(0, 72)
        elif self.task_level == "medium":
            initial_age = random.randint(0, 24)
        else:
            initial_age = random.randint(0, 6)

        return Bug(
            id=f"BUG-{index:03d}",
            title=title,
            severity=severity,
            bug_type=bug_type,
            status=BugStatus.OPEN,
            age_hours=initial_age,
            affected_users=affected_users,
            priority_score=priority_score,
            sla_hours=self.SLA_REQUIREMENTS[severity]
        )

    def generate_streaming_bug(self, step: int, base_id: int = 1000) -> Bug:
        bug = self._generate_single_bug(base_id + step)
        if random.random() < 0.4:
            bug.severity = random.choice([BugSeverity.HIGH, BugSeverity.CRITICAL])
            bug.priority_score = max(0.7, bug.priority_score)
            bug.sla_hours = self.SLA_REQUIREMENTS[bug.severity]
        return bug

    def _weighted_choice(self, choices, weights):
        total = sum(weights)
        normalized = [w / total for w in weights]
        return random.choices(choices, weights=normalized)[0]

    def get_bug_statistics(self, bugs: List[Bug]) -> Dict:
        if not bugs:
            return {}
        severity_counts = {s: 0 for s in BugSeverity}
        type_counts = {t: 0 for t in BugType}
        for bug in bugs:
            severity_counts[bug.severity] += 1
            type_counts[bug.bug_type] += 1
        return {
            "total_bugs": len(bugs),
            "severity_distribution": {k.value: v for k, v in severity_counts.items()},
            "type_distribution": {k.value: v for k, v in type_counts.items()},
            "avg_priority": sum(b.priority_score for b in bugs) / len(bugs),
            "total_affected_users": sum(b.affected_users for b in bugs),
            "critical_bugs": severity_counts[BugSeverity.CRITICAL]
        }