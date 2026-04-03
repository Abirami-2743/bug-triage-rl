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
---

# Bug Triage RL Environment

An OpenEnv-based RL environment for intelligent bug triage and escalation.

## API Endpoints
- `GET /` — Environment info
- `POST /reset` — Start new episode
- `POST /step` — Take action
- `GET /state` — Current state
- `GET /grade` — Episode score