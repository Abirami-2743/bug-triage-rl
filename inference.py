"""
Inference Script — Bug Triage RL Environment
Follows OpenEnv inference.py specification exactly.
"""

import os
import sys
import json
import requests
from typing import List, Optional
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN",     "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",  "https://abiraminayagi-bug-triage-rl.hf.space")

TASKS             = ["easy", "medium", "hard"]
BENCHMARK         = "bug-triage-rl"
MAX_STEPS         = 30
SUCCESS_THRESHOLD = 0.4
TEMPERATURE       = 0.1
MAX_TOKENS        = 300

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def env_reset(task_level):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_level": task_level, "max_steps": MAX_STEPS}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_level, action_type, bug_id, developer_id):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"task_level": task_level, "action_type": action_type, "bug_id": bug_id, "developer_id": developer_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade(task_level):
    r = requests.get(f"{ENV_BASE_URL}/grade", params={"task_level": task_level}, timeout=30)
    r.raise_for_status()
    return r.json()


def _heuristic_action(open_bugs, available_devs):
    if not open_bugs:
        return {"action_type": "close", "bug_id": "", "developer_id": None}
    open_bugs.sort(key=lambda b: b["priority_score"], reverse=True)
    top = open_bugs[0]
    if top["is_overdue"]:
        return {"action_type": "escalate", "bug_id": top["id"], "developer_id": None}
    if available_devs:
        available_devs.sort(key=lambda d: d["current_load"])
        return {"action_type": "assign", "bug_id": top["id"], "developer_id": available_devs[0]["id"]}
    if top["severity"] in ["critical", "high"]:
        return {"action_type": "escalate", "bug_id": top["id"], "developer_id": None}
    return {"action_type": "defer", "bug_id": top["id"], "developer_id": None}


def get_agent_action(observation, task_level, step):
    open_bugs = observation["bug_queue"]["open_bugs"]
    developers = observation["team"]["developers"]
    if not open_bugs:
        return {"action_type": "close", "bug_id": "", "developer_id": None}

    available_devs = [d for d in developers if d["is_available"]]
    bugs_text = "\n".join([f"- {b['id']}: {b['severity'].upper()} | {b['title']} | SLA: {b['sla_usage_pct']}% | {b['affected_users']} users" for b in open_bugs[:5]])
    devs_text = "\n".join([f"- {d['id']}: {d['name']} | {', '.join(d['skills'])} | Load: {d['current_load']}/{d['max_capacity']}" for d in available_devs[:4]]) if available_devs else "None available"

    prompt = f"""You are a software engineering manager doing bug triage.
Task: {task_level}, Step: {step}/{MAX_STEPS}

BUGS:
{bugs_text}

DEVELOPERS:
{devs_text}

Reply ONLY with JSON:
{{"action_type": "assign"|"escalate"|"defer"|"close", "bug_id": "<id>", "developer_id": "<id or null>"}}

Rules: ASSIGN critical bugs to available devs. ESCALATE overdue (SLA>100%). DEFER low priority. CLOSE resolved."""

    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        content = completion.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1] if "```json" not in content else content.split("```json")[1].split("```")[0]
        action = json.loads(content.strip())
        return {"action_type": action.get("action_type", "defer"), "bug_id": action.get("bug_id", open_bugs[0]["id"]), "developer_id": action.get("developer_id")}
    except Exception:
        return _heuristic_action(open_bugs, available_devs)


def run_task(task_level):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_level)
        observation = result["observation"]
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            action = get_agent_action(observation, task_level, step)
            action_str = f"{action['action_type']}(bug={action['bug_id']},dev={action['developer_id']})"
            error = None
            try:
                step_result = env_step(task_level, action["action_type"], action["bug_id"], action["developer_id"])
                observation = step_result["observation"]
                reward = float(step_result["reward"])
                done = bool(step_result["done"])
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            if done:
                break

        try:
            grade = env_grade(task_level)
            score = float(grade.get("score", 0.0))
        except Exception:
            score = sum(rewards) / max(len(rewards), 1)

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_level} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_level": task_level, "score": round(score, 4), "total_reward": round(sum(rewards), 4), "steps": steps_taken, "success": success}


def main():
    print("=" * 60, flush=True)
    print(f"Bug Triage RL — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME} | API: {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    results = []
    for task_level in TASKS:
        print(f"\n--- Task: {task_level.upper()} ---", flush=True)
        results.append(run_task(task_level))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    for r in results:
        print(f"{r['task_level'].upper():8} | score={r['score']:.4f} | {'PASS' if r['success'] else 'FAIL'}", flush=True)
    print(f"Avg score: {sum(r['score'] for r in results)/len(results):.4f}", flush=True)


if __name__ == "__main__":
    main()