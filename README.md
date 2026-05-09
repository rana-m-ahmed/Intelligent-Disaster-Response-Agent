
<h1 align="center">AIDRA</h1>
<p align="center"><b>Adaptive Intelligent Disaster Response Agent</b></p>
<p align="center">A hybrid AI orchestration platform for high-stakes urban rescue simulation and real-time decision support.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-Command%20Dashboard-000000?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Socket.IO-Real--Time%20Events-010101?logo=socketdotio&logoColor=white" alt="Socket.IO" />
  <img src="https://img.shields.io/badge/AI%2FML-Hybrid%20Reasoning-ff6f00" alt="AI ML" />
  <img src="https://img.shields.io/badge/Build-Experimental-orange" alt="Build Status" />
  <img src="https://img.shields.io/badge/License-Academic%20Use-blue" alt="License" />
</p>

<p align="center">
  <img width="900" src="https://capsule-render.vercel.app/api?type=waving&color=0:001F3F,100:005f73&height=120&section=header&text=AIDRA%20Mission%20Control&fontColor=ffffff&fontSize=30&animation=fadeIn" alt="AIDRA Header" />
</p>

---

## Executive Summary

AIDRA is a major multi-module Python AI system built for **disaster response in hazardous urban environments**. It combines search, optimization, constraint reasoning, machine learning, and fuzzy inference into one coordinated decision engine.

The platform continuously balances critical and conflicting objectives:

- Minimize rescue time.
- Avoid risk exposure.
- Prioritize high-severity victims.
- Allocate scarce resources under strict capacity constraints.

In evaluated scenarios, AIDRA achieved full rescue completion with strong operational KPIs, including zero risk-cell traversal for optimal plans and robust resource usage.

---

## 🧭 System Architecture (Hybrid AI Stack)

<p align="center">
  <img width="680" height="400" src="https://github.com/user-attachments/assets/6cd8ddb1-dd93-4e10-8131-ab4a5158385b" alt="AIDRA hybrid architecture diagram placeholder" />
</p>

<table align="center" width="100%">
  <tr>
    <td align="center"><b>Perception Layer</b><br/>Grid state, hazards, victims, resources</td>
    <td align="center"><b>Reasoning Layer</b><br/>Search + CSP + Fuzzy engine</td>
    <td align="center"><b>Prediction Layer</b><br/>Naive Bayes / kNN survival models</td>
    <td align="center"><b>Execution Layer</b><br/>Flask + Socket.IO real-time dashboard</td>
  </tr>
</table>

---

## 🧠 Core AI Modules

<p align="center">
  <b>Four tightly integrated AI pillars power AIDRA's adaptive behavior.</b>
</p>

<details open>
  <summary><b>1) Search & Planning</b> — A* + Risk-Aware Heuristics + Simulated Annealing</summary>

- Uses **A*** with domain-specific risk penalties to produce safe, high-quality routes.
- Benchmarked alternatives include **Greedy, BFS, and DFS**; A* delivered the most reliable outcomes in core tests.
- **Simulated Annealing** refines rescue ordering to improve mission-level efficiency.
- Practical impact: A* plans achieved optimal routing behavior with zero risk-cell traversal in key runs.

</details>

<details>
  <summary><b>2) Resource Allocation</b> — CSP with MRV + Forward Checking</summary>

- Models ambulance/team assignment under strict capacity limits.
- Uses **Minimum Remaining Values (MRV)** to prioritize constrained decisions.
- Uses **Forward Checking** to prune infeasible assignments early.
- Practical impact: backtracking reduced from **91 to 0** in reported evaluations.

</details>

<details>
  <summary><b>3) Survival Prediction (Machine Learning)</b> — Naive Bayes + kNN</summary>

- Predicts victim survival from injury severity, travel distance, elapsed time, and local risk exposure.
- Includes **Naive Bayes** and **k-Nearest Neighbors (kNN)** baselines.
- Practical impact: **Naive Bayes reached 91.2% accuracy**, outperforming kNN in the study.

</details>

<details>
  <summary><b>4) Uncertainty Handling</b> — Fuzzy Logic Urgency Engine</summary>

- Evaluates road blockage probability, hazard spread, and victim criticality.
- Produces a continuous **rescue urgency score** under uncertainty.
- Triggers real-time dynamic replanning when urgency exceeds threshold (e.g., > 70).

</details>

---

## 🖥️ Real-time Command Dashboard (Interactive UI)
<img width="1911" height="550" alt="image" src="https://github.com/user-attachments/assets/cc28ddc8-33ba-4b25-bf62-b0dafd9a1f11" />
<img width="1916" height="550" alt="image" src="https://github.com/user-attachments/assets/b22889bf-3ada-4201-9f0b-0cc48831ee61" />


AIDRA includes a **Flask + Socket.IO frontend** that acts as a live mission control surface for operators.

<table width="100%">
  <tr>
    <th align="left">UI Region</th>
    <th align="left">What You See</th>
    <th align="left">What It Does</th>
  </tr>
  <tr>
    <td><b>Left Panel</b><br/>KPI + Controls</td>
    <td>Victims saved, avg rescue time, risk exposure, path optimality, resource utilization, CSP stats, ML metrics</td>
    <td>Scenario switching, algorithm selection, risk-aversion tuning, pause/resume/reset, block mode, route planning</td>
  </tr>
  <tr>
    <td><b>Center Panel</b><br/>3D Urban Grid</td>
    <td>10x10 city map with safe/risk/blocked cells, med centers, rescue base, ambulances, victims</td>
    <td>Visualizes live routes, hazard spread, and dynamic interventions (blocked roads, new victims)</td>
  </tr>
  <tr>
    <td><b>Right Panel</b><br/>Decision Log</td>
    <td>Timestamped event stream</td>
    <td>Audit trail of user actions, planner decisions, and simulation events</td>
  </tr>
</table>

### Operator Interaction Model

- Click victim nodes to request route planning.
- Use **Block Mode** or modifier clicks to simulate road obstructions.
- Trigger dynamic victim events during simulation.
- Retrain ML metrics from the control panel.
- Observe planner adaptation in real time through route redraws and KPI updates.

<p>
  <kbd>Shift</kbd> + Click = block cell &nbsp; | &nbsp;
  <kbd>Ctrl</kbd> + Click = spawn victim &nbsp; | &nbsp;
  Click victim = route plan request
</p>

---

## 📊 Performance Matrix

<table width="100%" align="center">
  <tr>
    <th align="left">KPI</th>
    <th align="left">Reported Result</th>
    <th align="left">Notes</th>
  </tr>
  <tr>
    <td><b>Total Victims Rescued</b></td>
    <td>5 / 5</td>
    <td>Full completion in primary scenarios</td>
  </tr>
  <tr>
    <td><b>Average Rescue Time</b></td>
    <td>6.664 seconds</td>
    <td>Scenario-level aggregate</td>
  </tr>
  <tr>
    <td><b>Risk Exposure</b></td>
    <td>0 risk cells</td>
    <td>A* risk-aware planning benchmark runs</td>
  </tr>
  <tr>
    <td><b>Resource Utilization</b></td>
    <td>76% - 85%</td>
    <td>Ambulance/team/kit usage efficiency window</td>
  </tr>
  <tr>
    <td><b>ML Accuracy (Naive Bayes)</b></td>
    <td>91.2%</td>
    <td>Best-performing survival predictor</td>
  </tr>
  <tr>
    <td><b>CSP Backtracking</b></td>
    <td>91 -> 0</td>
    <td>MRV + Forward Checking effectiveness</td>
  </tr>
</table>

---

## 🚀 Installation & Usage

### 1) Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Intelligent-Disaster-Response-Agent
```

### 2) Create Virtual Environment & Install Dependencies

```bash
cd aidra
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> On Windows PowerShell, activate with: `.venv\\Scripts\\Activate.ps1`

### 3) Run Flask + SocketIO Dashboard

```bash
python app.py
```

Open the app in your browser at `http://127.0.0.1:5000`.

### 4) Optional Offline Scenario Runner

```bash
python main.py
```

### 5) Run Tests (if configured)

```bash
python -m pytest
```

---

## 📁 Repository Structure

```text
Intelligent-Disaster-Response-Agent/
├── README.md                    # This landing page
└── aidra/
    ├── app.py                   # Flask + SocketIO real-time app
    ├── main.py                  # Compatibility shim / runner entry
    ├── src/                     # Core AI modules (search, csp, fuzzy, ml)
    ├── static/                  # Frontend JS/CSS + Three.js controls
    ├── templates/               # HTML dashboard template
    ├── tests/                   # Test suite
    ├── models/                  # Trained model metrics artifacts
    ├── logs/                    # Decision and runtime logs
    └── results/                 # KPI and evaluation outputs
```

---

## 👥 Contributors & Acknowledgements

<p align="center">
  <b>Rana Muhammad Ahmed</b> (01-134241-039)<br/>
  <b>Sabahat</b> (01-134241-041)<br/>
  Department of Computer Science<br/>
  Bahria University Islamabad Campus
</p>

<p align="center">
  Built as an academic flagship project integrating modern AI techniques for mission-critical rescue planning.
</p>

---

## 📌 Citation-Friendly Project Descriptor

**AIDRA (Adaptive Intelligent Disaster Response Agent)** is a hybrid AI disaster-response platform that integrates risk-aware search, CSP-based resource allocation, survival prediction, and fuzzy uncertainty reasoning in a real-time Flask/Socket.IO command dashboard for dynamic urban rescue simulation.
