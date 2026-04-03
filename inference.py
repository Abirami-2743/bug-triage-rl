"""
Baseline Inference Script for Bug Triage RL Environment
Uses OpenAI API client to run LLM agent against the environment
Emits structured [START] / [STEP] / [END] logs as required by OpenEnv spec

Usage:
    API_BASE_URL=https://your-hf-space.hf.space \
    MODEL_NAME=gpt-4o-mini \
    OPENAI_API_KEY=your-key \
    python inference.py
"""

import os
import sys
import json
import time
from openai import OpenAI

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from server.bug_triage_environment import BugTriageEnvironment

# Environment variables (required by OpenEnv spec)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Use HF_TOKEN as API key if OPENAI_API_KEY not set
api_key = OPENAI_API_KEY or HF_TOKEN or "dummy-key"

# Initialize OpenAI client (as required by spec)
client = OpenAI(
    api_key=api_key,
    base_url=API_BASE_URL
)

TASK_LEVELS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 50


def get_agent_action(observation: dict, task_level: str) -> dict:
    """
    Use LLM to decide the next action based on current observation
    Returns action dict with action_type, bug_id, developer_id
    """
    open_bugs = observation["bug_queue"]["open_bugs"]
    developers = observation["team"]["developers"]

    if not open_bugs:
        return {"action_type": "close", "bug_id": "", "developer_id": None}

    # Build prompt for LLM
    bugs_summary = "\n".join([
        f"- {b['id']}: {b['severity'].upper()} | {b['title']} | SLA: {b['sla_usage_pct']}% used | {b['affected_users']} users"
        for b in open_bugs[:5]
    ])

    available_devs = [d for d in developers if d["is_available"]]
    devs_summary = "\n".join([
        f"- {d['id']}: {d['name']} | Skills: {', '.join(d['skills'])} | Load: {d['current_load']}/{d['max_capacity']}"
        for d in available_devs[:4]
    ]) if available_devs else "No developers available"

    prompt = f"""You are an expert software engineering manager doing bug triage.

Task level: {task_level}
Current step: {observation['metrics']['step']}/{observation['metrics']['max_steps']}

BUG QUEUE (top priority bugs):
{bugs_summary}

AVAILABLE DEVELOPERS:
{devs_summary}

Choose ONE action. Respond ONLY with valid JSON:
{{
  "action_type": "assign" | "escalate" | "defer" | "close",
  "bug_id": "<bug id from the list above>",
  "developer_id": "<developer id if assigning, else null>",
  "reasoning": "<brief reason>"
}}

Rules:
- ASSIGN critical/high bugs to available developers with matching skills
- ESCALATE overdue bugs (SLA > 100%)
- DEFER low priority bugs when team is full
- CLOSE bugs that are resolved or invalid
- Never defer critical bugs"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()

        # Clean up response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        action = json.loads(content)

        # Validate required fields
        if "action_type" not in action or "bug_id" not in action:
            raise ValueError("Missing required fields")

        return {
            "action_type": action.get("action_type", "defer"),
            "bug_id": action.get("bug_id", open_bugs[0]["id"]),
            "developer_id": action.get("developer_id", None)
        }

    except Exception as e:
        # Fallback to heuristic if LLM fails
        return _heuristic_action(open_bugs, available_devs)


def _heuristic_action(open_bugs: list, available_devs: list) -> dict:
    """Fallback heuristic agent"""
    if not open_bugs:
        return {"action_type": "close", "bug_id": "", "developer_id": None}

    # Sort by priority
    open_bugs.sort(key=lambda b: b["priority_score"], reverse=True)
    top_bug = open_bugs[0]

    if top_bug["is_overdue"]:
        return {"action_type": "escalate", "bug_id": top_bug["id"], "developer_id": None}

    if available_devs:
        available_devs.sort(key=lambda d: d["current_load"])
        return {
            "action_type": "assign",
            "bug_id": top_bug["id"],
            "developer_id": available_devs[0]["id"]
        }

    if top_bug["severity"] in ["critical", "high"]:
        return {"action_type": "escalate", "bug_id": top_bug["id"], "developer_id": None}

    return {"action_type": "defer", "bug_id": top_bug["id"], "developer_id": None}


def run_task(task_level: str) -> dict:
    """Run one full task episode and return results"""

    env = BugTriageEnvironment(task_level=task_level, max_steps=MAX_STEPS_PER_TASK)
    observation = env.reset()

    # [START] log — required by OpenEnv spec
    start_log = {
        "type": "START",
        "task_level": task_level,
        "episode_id": env.episode_id,
        "total_bugs": observation["bug_queue"]["total_count"],
        "team_size": len(observation["team"]["developers"]),
        "timestamp": time.time()
    }
    print(f"[START] {json.dumps(start_log)}")

    total_reward = 0.0
    steps = 0

    while not env.done and steps < MAX_STEPS_PER_TASK:
        # Get action from LLM agent
        action = get_agent_action(observation, task_level)

        # Execute action
        result = env.step(action)
        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        total_reward += reward
        steps += 1

        # [STEP] log — required by OpenEnv spec
        step_log = {
            "type": "STEP",
            "task_level": task_level,
            "step": steps,
            "action_type": action["action_type"],
            "bug_id": action["bug_id"],
            "developer_id": action.get("developer_id"),
            "reward": reward,
            "cumulative_reward": round(total_reward, 4),
            "open_bugs": info.get("open_bugs", 0),
            "done": done
        }
        print(f"[STEP] {json.dumps(step_log)}")

        if done:
            break

    # Calculate final score
    score = env.grade()

    # [END] log — required by OpenEnv spec
    end_log = {
        "type": "END",
        "task_level": task_level,
        "episode_id": env.episode_id,
        "total_steps": steps,
        "total_reward": round(total_reward, 4),
        "score": round(score, 4),
        "bugs_processed": env.episode_metrics.total_bugs_processed,
        "resolved_within_sla": env.episode_metrics.bugs_resolved_within_sla,
        "timestamp": time.time()
    }
    print(f"[END] {json.dumps(end_log)}")

    return {
        "task_level": task_level,
        "score": round(score, 4),
        "total_reward": round(total_reward, 4),
        "steps": steps
    }


def main():
    print("=" * 60)
    print("Bug Triage RL Environment — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("=" * 60)

    results = []

    for task_level in TASK_LEVELS:
        print(f"\nRunning task: {task_level.upper()}")
        print("-" * 40)

        try:
            result = run_task(task_level)
            results.append(result)
            print(f"Task {task_level}: score={result['score']}, reward={result['total_reward']}, steps={result['steps']}")
        except Exception as e:
            print(f"ERROR on task {task_level}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_level": task_level,
                "score": 0.0,
                "total_reward": 0.0,
                "steps": 0,
                "error": str(e)
            })

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["score"] >= 0.4 else "FAIL"
        print(f"{r['task_level'].upper():8} | score={r['score']:.4f} | reward={r['total_reward']:.4f} | {status}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()