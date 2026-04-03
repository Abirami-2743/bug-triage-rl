"""
OpenEnv FastAPI Server for Bug Triage Environment
Exposes step() / reset() / state() via HTTP endpoints
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from server.bug_triage_environment import BugTriageEnvironment

app = FastAPI(
    title="Bug Triage RL Environment",
    description="OpenEnv-based RL environment for intelligent bug triage — OpenEnv Hackathon 2026",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment instances per task level
envs: Dict[str, BugTriageEnvironment] = {}


class ResetRequest(BaseModel):
    task_level: str = "medium"
    max_steps: int = 100


class StepRequest(BaseModel):
    task_level: str = "medium"
    action_type: str  # assign / escalate / defer / close
    bug_id: str
    developer_id: Optional[str] = None


class ActionRequest(BaseModel):
    action_type: str
    bug_id: str
    developer_id: Optional[str] = None


def get_env(task_level: str) -> BugTriageEnvironment:
    if task_level not in envs:
        envs[task_level] = BugTriageEnvironment(task_level=task_level)
    return envs[task_level]


@app.get("/")
def root():
    return {
        "name": "Bug Triage RL Environment",
        "version": "0.1.0",
        "description": "OpenEnv-based RL environment for intelligent bug triage",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
        "tasks": ["easy", "medium", "hard"],
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset environment and start new episode"""
    if request.task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(400, "task_level must be easy, medium, or hard")

    env = BugTriageEnvironment(
        task_level=request.task_level,
        max_steps=request.max_steps
    )
    envs[request.task_level] = env
    observation = env.reset()

    return {
        "observation": observation,
        "task_level": request.task_level,
        "episode_id": env.episode_id
    }


@app.post("/step")
def step(request: StepRequest):
    """Take one step in the environment"""
    env = get_env(request.task_level)

    result = env.step({
        "action_type": request.action_type,
        "bug_id": request.bug_id,
        "developer_id": request.developer_id
    })

    return result


@app.get("/state")
def state(task_level: str = "medium"):
    """Get current episode state"""
    env = get_env(task_level)
    return env.state()


@app.get("/grade")
def grade(task_level: str = "medium"):
    """Get episode performance score 0.0–1.0"""
    env = get_env(task_level)
    score = env.grade()
    return {
        "task_level": task_level,
        "score": round(score, 4),
        "cumulative_reward": round(env.cumulative_reward, 4),
        "steps_taken": env.current_step,
        "bugs_processed": env.episode_metrics.total_bugs_processed,
        "resolved_within_sla": env.episode_metrics.bugs_resolved_within_sla
    }


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions"""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Simple bugs, relaxed SLAs, full team available",
                "max_bugs": 8,
                "team_size": 3,
                "passing_score": 0.5
            },
            {
                "name": "medium",
                "description": "Mixed severity, some SLA pressure, partial team",
                "max_bugs": 15,
                "team_size": 4,
                "passing_score": 0.6
            },
            {
                "name": "hard",
                "description": "Critical bugs, SLA overdue, overwhelmed team, crisis events",
                "max_bugs": 25,
                "team_size": 8,
                "passing_score": 0.7
            }
        ]
    }


# Also expose flat API for frontend
@app.post("/env/init")
def env_init(request: ResetRequest):
    return reset(request)


@app.post("/env/reset")
def env_reset(request: ResetRequest):
    return reset(request)


@app.get("/env/state")
def env_state(task_level: str = "medium"):
    env = get_env(task_level)
    state_data = env.state()
    obs = env._get_observation()
    state_data["bugs"] = obs["bug_queue"]["open_bugs"]
    state_data["developers"] = obs["team"]["developers"]
    state_data["queue_health"] = round(env.bug_queue.queue_health_score * 100, 1)
    state_data["team_availability"] = round(env.team_state.availability_ratio * 100, 1)
    state_data["episode_running"] = not env.done
    state_data["cumulative_reward"] = round(env.cumulative_reward, 2)
    return state_data


@app.post("/env/manual_action")
def env_manual_action(task_level: str = "medium", request: ActionRequest = None):
    env = get_env(task_level)
    result = env.step({
        "action_type": request.action_type,
        "bug_id": request.bug_id,
        "developer_id": request.developer_id
    })
    result["is_good_decision"] = result["reward"] > 0
    result["message"] = result["info"].get("reward_reason", "")
    return result


@app.post("/env/ai_step")
def env_ai_step(task_level: str = "medium"):
    """Heuristic AI agent makes one decision"""
    env = get_env(task_level)

    open_bugs = [b for b in env.bug_queue.bugs if b.status.value == "open"]
    if not open_bugs:
        return {"message": "No open bugs", "done": True}

    open_bugs.sort(key=lambda b: b.priority_score, reverse=True)
    top_bug = open_bugs[0]

    available_devs = [d for d in env.team_state.developers if d.is_available]

    if top_bug.age_hours >= top_bug.sla_hours:
        action_type = "escalate"
        dev_id = None
    elif available_devs:
        available_devs.sort(key=lambda d: d.current_load)
        action_type = "assign"
        dev_id = available_devs[0].id
    elif top_bug.severity.value in ["critical", "high"]:
        action_type = "escalate"
        dev_id = None
    else:
        action_type = "defer"
        dev_id = None

    result = env.step({
        "action_type": action_type,
        "bug_id": top_bug.id,
        "developer_id": dev_id
    })

    result["action_taken"] = action_type
    result["bug_id"] = top_bug.id
    result["bug_severity"] = top_bug.severity.value
    return result


@app.get("/env/summary")
def env_summary(task_level: str = "medium"):
    env = get_env(task_level)
    score = env.grade()
    if score >= 0.8:
        performance = "Excellent"
    elif score >= 0.6:
        performance = "Good"
    elif score >= 0.4:
        performance = "Average"
    else:
        performance = "Poor"

    return {
        "total_reward": round(env.cumulative_reward, 2),
        "total_steps": env.current_step,
        "bugs_processed": env.episode_metrics.total_bugs_processed,
        "resolved_within_sla": env.episode_metrics.bugs_resolved_within_sla,
        "success_score": round(score, 2),
        "performance": performance,
        "task_level": task_level
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)