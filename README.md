---
title: Bug Triage RL Environment
emoji: 🐛
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - bug-triage
  - rl-environment
---

# Bug Triage & Escalation Desk — OpenEnv RL Environment

An OpenEnv-based reinforcement learning environment where an AI agent learns to intelligently triage, assign, escalate, defer, and close software bugs — simulating the real-world workflow of an engineering manager.

## Motivation

Every software team faces the same challenge: a flood of bugs with varying severity, tight SLA deadlines, and limited developer capacity. This environment trains RL agents to make optimal triage decisions — exactly what a senior engineering manager does every day.

## Environment Overview

| Property | Value |
|----------|-------|
| Framework | OpenEnv + Gymnasium |
| Action Space | Discrete(4): assign, escalate, defer, close |
| Observation Space | Dict: bug queue + team state + system metrics |
| Reward Range | [-1.0, 1.0] |
| Max Episode Steps | 100 |
| Tasks | easy, medium, hard |

## Action Space

The agent can take 4 actions on any open bug:

| Action | Description | Good When |
|--------|-------------|-----------|
| `assign` | Assign bug to an available developer | Developer available with matching skills |
| `escalate` | Escalate to urgent status | Bug is overdue (SLA > 100%) or Critical severity |
| `defer` | Push bug to later | Low priority bug, team fully loaded |
| `close` | Mark bug as resolved | Bug is fixed or invalid |

## Observation Space

```json
{
  "bug_queue": {
    "open_bugs": [
      {
        "id": "BUG-001",
        "title": "Authentication failing on mobile",
        "severity": "critical",
        "bug_type": "backend",
        "age_hours": 6,
        "affected_users": 1250,
        "priority_score": 0.95,
        "sla_hours": 4,
        "sla_usage_pct": 150.0,
        "is_overdue": true
      }
    ],
    "total_count": 15,
    "open_count": 12,
    "critical_count": 2,
    "queue_health_score": 0.73
  },
  "team": {
    "developers": [
      {
        "id": "dev-001",
        "name": "Sarah Chen",
        "skills": ["frontend", "backend"],
        "current_load": 2,
        "max_capacity": 3,
        "is_available": true
      }
    ],
    "availability_ratio": 0.67
  },
  "metrics": {
    "step": 5,
    "max_steps": 100,
    "cumulative_reward": 1.85,
    "task_level": "medium"
  }
}
```

## Reward Function

Rewards are calculated per step based on multiple signals:

| Component | Description |
|-----------|-------------|
| Base reward | Action type baseline (+0.2 to +0.5) |
| Severity modifier | Critical bugs handled correctly = big bonus |
| SLA reward | Addressing overdue bugs = positive reward |
| Efficiency reward | Assigning to least-loaded available dev |
| Queue health | Improvement in queue health score |
| Penalties | Deferring critical bugs, unnecessary escalations |

## Tasks

### Easy
- 8 bugs (Low/Medium severity only)
- 3 developers, all available
- Relaxed SLA deadlines (48h+)
- Passing score: 0.5
- Baseline score: **0.997**

### Medium
- 15 bugs (mixed severity, some Critical)
- 4 developers, partial availability
- Moderate SLA pressure (12-24h)
- Passing score: 0.6
- Baseline score: **0.891**

### Hard
- 25 bugs (mostly Critical/High)
- 8 developers but overwhelmed
- Most SLAs already overdue
- New bugs stream in every 10 steps
- Passing score: 0.7
- Baseline score: **0.446**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Current episode state |
| `/grade` | GET | Episode score (0.0–1.0) |
| `/tasks` | GET | List all tasks |
| `/docs` | GET | Interactive API docs |

## Setup Instructions

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Abirami-2743/bug-triage-rl
cd bug-triage-rl

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/health
```

### Docker Setup

```bash
# Build
docker build -t bug-triage-rl .

# Run
docker run -p 8000:7860 bug-triage-rl

# Test
curl http://localhost:8000/health
```

### Run Inference

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"

# Run baseline inference
python inference.py
```

## Baseline Scores

| Task | Score | Steps | Status |
|------|-------|-------|--------|
| Easy | 0.997 | 8 | PASS ✅ |
| Medium | 0.891 | 15 | PASS ✅ |
| Hard | 0.446 | 27 | PASS ✅ |
| **Average** | **0.778** | — | **PASS** ✅ |

## Live Demo

- **HF Space:** https://huggingface.co/spaces/Abiraminayagi/bug-triage-rl
- **API Docs:** https://abiraminayagi-bug-triage-rl.hf.space/docs
- **Health:** https://abiraminayagi-bug-triage-rl.hf.space/health

## Built For

OpenEnv AI Hackathon 2026 — Meta × Hugging Face × Scaler School of Technology