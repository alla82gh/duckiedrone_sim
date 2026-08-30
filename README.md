# or use any editor

# Look for lines like:
# <<<<<<< HEAD
# Your local content
# =======
# Remote content
# >>>>>>> <commit-hash>

# Edit to keep what you want, then save.

# After resolving, add the file:
git add README.md

# Complete the merge:
git commit -m "Merge remote changes"# Duckiedrone DD21 Simulation and Control Framework

ROS Noetic + Gazebo simulation framework for the **Duckiedrone DD21** quadrotor, developed for research on system identification, model-based control, uncertainty-aware modeling, and UAV validation.

## Project Scope

This repository supports the experimental workflow used to compare several quadrotor control and modeling approaches on a common simulation platform:

- Cascaded PID control
- Physics-based Model Predictive Control (Physics MPC)
- Prediction Error Method based MPC (PEM-MPC)
- Hybrid PEM-Gaussian Process MPC (Hybrid PEM-GP MPC)
- Validation scenarios and performance logging
- Future transfer to the real Duckiedrone DD21 platform

## Platform

- Ubuntu 20.04
- ROS Noetic
- Gazebo 11
- Python 3
- Duckiedrone DD21 quadrotor model
- Catkin workspace

Main ROS packages are located under:

```text
src/
```

The current validation framework is primarily developed under:

```text
src/duckiedrone_validation/
```

## Controller Architecture

The current controller development sequence is:

```text
PID -> Physics MPC -> PEM-MPC -> Hybrid PEM-GP MPC -> Real Drone
```

### Physics MPC

The Physics MPC implementation uses a discrete linear prediction model and a quadratic-programming formulation.

Current nominal configuration:

```text
State dimension       nx = 12
Input dimension       nu = 4
Sampling time         Ts = 0.01 s
Prediction horizon    Np = 20
Control horizon       Nc = 20
QP solver             OSQP
```

The controller includes:

- state prediction matrices
- quadratic tracking cost
- input constraints
- control-increment constraints
- rotor-feasibility constraints
- soft state constraints using slack variables

## Validation Scenarios

The validation framework includes scenarios such as:

- **S1** - Hovering
- **S2-Roll** - Roll attitude step test
- **S2-Pitch** - Pitch attitude step test
- **S2-Yaw** - Yaw attitude step test
- **S3** - Circular trajectory tracking
- **S6** - Point tracking

Additional disturbance and model-mismatch scenarios are under development.

## Graphical Interface

A PyQt5 graphical interface is available for launching Gazebo, selecting controllers, selecting validation scenarios, and starting/stopping experiments.

Main script:

```text
src/duckiedrone_validation/scripts/gui.py
```

Run it from the catkin workspace after sourcing ROS:

```bash
cd ~/duckiedrone_sim
source devel/setup.bash
python3 src/duckiedrone_validation/scripts/gui.py
```

## Building the Workspace

From the repository root:

```bash
cd ~/duckiedrone_sim
catkin_make
source devel/setup.bash
```

To source the workspace automatically in a new terminal, you may add the following line to `~/.bashrc`:

```bash
source ~/duckiedrone_sim/devel/setup.bash
```

## Running the Simulation

The exact launch command depends on the active validation package and scenario configuration. A typical workflow is:

```bash
cd ~/duckiedrone_sim
source devel/setup.bash
roslaunch duckiedrone_validation spawn_dd21.launch
```

Then run the required controller/scenario through the validation launch files or the GUI.

## Repository Structure

```text
duckiedrone_sim/
├── src/                         # ROS source packages
├── build/                       # Catkin build output
├── devel/                       # Catkin development space
├── .vscode/                     # VS Code configuration
├── .catkin_workspace
└── README.md
```

Important validation code is organized approximately as:

```text
src/duckiedrone_validation/
├── launch/
├── scripts/
│   ├── controllers/
│   │   └── mpc/
│   ├── models/
│   ├── scenarios/
│   └── gui.py
├── config/
└── data/
```

## Research Context

This repository is part of ongoing research on identification, modeling, control, and uncertainty-aware prediction for quadrotor UAVs. The simulation framework is designed to keep the same vehicle model, scenarios, logging structure, and evaluation metrics across controllers so that comparisons remain reproducible.

Related system-identification work includes a hybrid Prediction Error Method and Gaussian Process framework for uncertainty-aware quadrotor modeling.

## Development Status

Current status:

- PID baseline validated
- Physics MPC implemented and under scenario-level validation
- PEM artifacts and model integration under development
- Hybrid PEM-GP uncertainty model integration under development
- Real-platform validation planned after simulation validation

## Author

**Abdallah GHOUL**  
Researcher in quadrotor system identification, modeling, control, and teleoperation.
## Email: 
abdallah.ghoul@univ-bechar.dz
