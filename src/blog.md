# 🐛 Bug Triage RL: Teaching a 1.5B Model to Think Like a Senior Engineering Manager

**OpenEnv AI Hackathon 2026 — Meta × Hugging Face × Scaler**
**Team FYA.dev**

> **TL;DR:** We built a high-fidelity DevOps simulation where an LLM must triage a live bug queue under real operational pressure. Using GRPO + Unsloth on a 1.5B model, the agent went from randomly deferring Critical bugs to correctly escalating overdue SLA breaches — a fundamental behavioral shift, not just better numbers.

---

## The Problem No One Has Benchmarked

It's 2 AM. Production is down. 25 bugs in the queue. 8 developers — half at capacity, some skill-mismatched. Most SLAs already breached. Slack is on fire.

**This is bug triage under pressure.** Every engineering team — from early startups to Meta — does this manually, every single day. A wrong call cascades: a deferred Critical bug becomes a full outage; an overloaded developer misses three deadlines; a queue that looks manageable at 9 AM is unmanageable by noon.

No standardized RL benchmark exists for this. We built one.

---

## The Environment

Bug Triage RL is a **sequential decision-making environment** built on OpenEnv + FastAPI + Gymnasium. At each step the agent sees a live bug queue and a developer team, then acts.


Bug Queue (dynamic)  +  Developer Team  →  Agent  →  Action
        ↓                                               ↓
    SLA Engine                              Reward Calculator


| Property | Value |
|----------|-------|
| Framework | OpenEnv + FastAPI + Gymnasium |
| Actions | assign · escalate · defer · close |
| Observation | Bug queue + team state + system metrics |
| Reward range | [−1.0, +1.0] — 5-component composite |
| Hard mode | 25 bugs, 8 devs overwhelmed, streaming arrivals every 10 steps |

### Why This Is Genuinely Hard

This isn't "pick the right label." It's a **sequential problem with cascading consequences**:

- Assigning your best backend dev now means she's unavailable for the next Critical bug
- New bugs stream in during Hard mode — static strategies fail instantly
- A bug at 95% SLA is fundamentally different from one at 156% SLA — urgency is continuous
- You can't assign a frontend dev to a database deadlock — skills and capacity both matter

### Reward Design That Can't Be Gamed

5 independent reward components — the agent cannot exploit a single shortcut:

| Component | What It Teaches |
|-----------|----------------|
| Base reward | Action type selection |
| Severity modifier | Critical bugs demand immediate action — deferring them is catastrophic |
| SLA reward | Urgency scales continuously with deadline proximity |
| Efficiency reward | Balance workload — don't overload one dev while others are idle |
| Queue health Δ | Systemic improvement across the full episode |

**Episode Score = SLA Compliance × 0.4 + Queue Health × 0.3 + Team Utilization × 0.2 + Efficiency × 0.1**

---

## Training

We used **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL with **Unsloth** for 2–5× memory efficiency — running entirely on a free Kaggle T4 GPU.

| Detail | Value |
|--------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Method | GRPO (no value model needed) |
| LoRA | r=16, lora_alpha=32 |
| Temperature | 1.8 (diverse rollouts for GRPO signal) |
| Steps | 500 on Hard difficulty |
| Hardware | Kaggle T4 (free) |

GRPO reward = environment reward + format bonus (+0.3 valid JSON) + action validity (+0.2)

📓 **Full notebook (re-runnable):** [kaggle.com/code/brindapalanimuthu/bug-triage-final](https://www.kaggle.com/code/brindapalanimuthu/bug-triage-final)

---

## Results

![GRPO Training Results — reward curve and before/after scores](grpo_training_results.png)
*Left: Training reward curve over 500 steps (Hard mode) — rolling average improves from −0.40 to −0.30. Right: Normalized episode scores before and after GRPO — all tiers pass the threshold.*

The reward curve shows the agent moving from noisy, random decisions (−0.40) toward more consistent triage behavior (−0.30) over 500 steps. The before/after scores (0.951 → 0.941) are comparable — indicating the **1.5B base model was already strong on Hard tier**. The real evidence of learning is in behavior, not just scores.

---

## The Real Proof: Before vs. After Behavior

Same bug. Same state. Completely different decisions.

**Situation:** `BUG-001` — "Authentication failing on mobile"
- Severity: **CRITICAL**
- SLA: **156% overdue** (breach already in progress)
- Users affected: **1,250**

**Before GRPO:**
```json
{"action_type": "defer", "bug_id": "BUG-001", "developer_id": null}
```
❌ Catastrophic. Deferring a Critical bug at 156% SLA triggers the harshest penalty in the reward function — and in real life, means 1,250 users stay blocked while the queue continues to deteriorate.

**After GRPO:**
```json
{"action_type": "escalate", "bug_id": "BUG-001", "developer_id": null}
```
✅ Correct. SLA breached + Critical severity = immediate escalation. The agent learned this policy purely through interaction with the environment — not from rules, not from labels.

**This is the core result.** The agent learned *when deferral is safe and when it is catastrophic* — a distinction that requires simultaneously reasoning about severity, SLA status, and queue health. That's not memorization; bugs are randomly generated each episode.

---

## 🎬 Live Demo

**Try it yourself — no setup needed:**

👉 **[huggingface.co/spaces/Abiraminayagi/bug-triage-rl/docs](https://abiraminayagi-bug-triage-rl.hf.space/docs)**

### Run a full episode in your browser:

**Step 1 — Start a Hard mode episode:**
```
POST /reset  →  {"task_level": "hard"}
```

**Step 2 — Take an action:**
```
POST /step  →  {"action_type": "escalate", "bug_id": "BUG-001", "developer_id": null}
```

**Step 3 — Check your score:**
```
GET /grade
```

The live Swagger UI at `/docs` lets you run all of this interactively in the browser — no code, no setup, no API key needed. Click, submit, see the reward.

---

## Why This Matters

Bug triage is a **daily, high-stakes, high-frequency decision** at every software company. The cost of getting it wrong is measured in outages, SLA penalties, and engineer burnout. Yet no RL benchmark exists for it.

Our environment fills that gap:
- **Realistic complexity** — not a toy; Hard tier simulates active production incidents
- **Informative reward signals** — 5 components, not binary success/failure
- **Difficulty scaling** — Easy to Hard, covering routine triage to full crisis management
- **Open infrastructure** — fully OpenEnv compliant, Gymnasium compatible, Docker deployable

A compact fine-tuned model running as a real-time triage assistant — embedded in Jira or GitHub Issues, handling the 80% of decisions that are straightforward — is not a distant future. This environment is the benchmark to get there.

---

## Links

| Resource | Link |
|----------|-------|
| 🚀 Live Environment | [huggingface.co/spaces/Abiraminayagi/bug-triage-rl](https://huggingface.co/spaces/Abiraminayagi/bug-triage-rl) |
| 📓 Training Notebook | [kaggle.com/code/brindapalanimuthu/bug-triage-final](https://www.kaggle.com/code/brindapalanimuthu/bug-triage-final) |
| 🧠 Trained Model | [huggingface.co/Abiraminayagi/bug-triage-grpo-trained](https://huggingface.co/Abiraminayagi/bug-triage-grpo-trained) |
| 💻 GitHub | [github.com/Abirami-2743/bug-triage-rl](https://github.com/Abirami-2743/bug-triage-rl) |

---

*Built with ❤️ by Team FYA.dev — OpenEnv AI Hackathon 2026*
*We didn't just train a model. We built the benchmark that didn't exist.*
