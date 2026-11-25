# Start Here — Antigravity, step by step

## Before you start

1. Install Antigravity and open this entire folder as the workspace (not just `PROMPTS.md`).
2. Use Python 3.11 or 3.12 and Node.js 20 LTS. You do not need to buy a radar board tonight; this is a software simulation.
3. Keep `PLAN.md`, `INTERFACES.md`, `radar_config.yaml`, `scene.json`, `material_db.yaml`, and `motion_config.yaml` in the workspace. They are the shared project contract.
4. (Recommended) Open a terminal and run `git init` so you can review/revert each task. Commit after a task passes.
5. Do not let multiple agents edit the same task files simultaneously. Run tasks in order.

## How to give tasks to Antigravity

1. Open `PROMPTS.md`.
2. Copy the **Shared instruction block** and the complete prompt for `T0` into one Antigravity agent session.
3. Let the agent inspect the repository, implement only T0, run its tests, and report what passed or failed.
4. Review the changed files and run the stated commands. Do not continue if T0 has unresolved errors.
5. Repeat with T1, then T2, through T7. Each task is designed to depend on the previous task's interfaces.
6. If Antigravity suggests changing an interface, ask it to update `INTERFACES.md`, the producer, the consumer and contract tests together. Do not accept undocumented changes.

## Suggested order for tonight

| Order | Task | What you should see at the end |
|---|---|---|
| 1 | T0 — Bootstrap | Valid configuration, derived radar metrics, runnable Python package |
| 2 | T1 — Room | 3D room with bed, sofa, table, chair, wardrobe, door and window |
| 3 | T2 — Human | Fully visible human avatar entering and performing randomized actions |
| 4 | T3 — RF | Direct/reflected paths, occlusion and path-strength output |
| 5 | T4 — Radar | Complex FMCW ADC cube with configurable impairments |
| 6 | T5 — DSP | Range-Doppler, detections, point cloud and basic fall-event baseline |
| 7 | T6 — Dashboard | Four synchronized panels and useful controls |
| 8 | T7 — Integration | Fixed-seed demo, tests, logs and a report of limitations |

If the night becomes short, prioritize a correct end-to-end thin slice: one scene, one avatar, one direct/reflected path, a small valid ADC cube, one DSP result and all four panels synchronized. Do not fake a result to make the dashboard look complete.

## Windows setup

After T0 has created the Python package, open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

The script checks Python, creates `.venv`, installs `requirements.txt`, and installs frontend packages if the UI package has been created.

Run the backend in PowerShell window 1:

```powershell
.
un_backend.ps1
```

Run the frontend in PowerShell window 2:

```powershell
.
un_frontend.ps1
```

Open the local URL printed by Vite (normally `http://localhost:5173`). If a script reports that the task has not created a required file yet, finish the relevant Antigravity task first.

## What “done tonight” means

A successful tonight prototype means the room, avatar, randomized motion, RF-path model, waveform/ADC model, DSP outputs and synchronized dashboard run together for fixed seeds. It does **not** mean 95% real-world fall recall, accurate human RCS, camera-like 3D reconstruction, or physical-hardware equivalence has been proven. Those require measured hardware data and a separate validation campaign.
