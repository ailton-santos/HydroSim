# HydroSim: Hydraulic Modeling & Optimization Tools

🚧 **Note: This project is currently under active implementation (Work in Progress). Features and API structures are subject to change.** 🚧

**HydroSim** is a robust Python tool developed for the control, calibration, and parametric analysis of two-dimensional hydraulic simulation models. It acts as a high-level interface capable of orchestrating simulation and optimization runs for established industry software, such as **SRH-2D**, **HEC-RAS**, and **Backwater-1D** models.

## 🚀 Key Features

* **Simulation Engine Agnostic:** Standardized and integrated support for SRH-2D and HEC-RAS 2D.
* **Automated Calibration (`Calibrator`):** Evaluate calibration parameters (e.g., Manning's roughness coefficients) using local or global optimizers (supported via `scipy.optimize`), comparing outputs (VTK, HDF) against real-world field measurement data.
* **Parametric Studies (`Parametric_Study`):** Execute sampling across a parameter space (using strategies like Latin Hypercube via `skopt`) to automatically create tens or hundreds of isolated hydraulic scenarios.
* **Clean Data Management:** Extensive support for handling mesh and geometry files (`.srhgeom`, `.srhmat`, `.hdf`), smart extraction of VTK results, and `.csv` export for statistical reporting.

## 📦 Module Structure

* **`Calibrator.py` & `Objectives.py`**: Manages the JSON configuration and orchestrates successive runs to minimize the difference (Error Score) between simulated results and measured sampling points.
* **`Optimizer.py`**: A wrapper for `scipy.optimize` algorithms and custom sweeps (*enumerator*), keeping track of execution and convergence history.
* **`Parametric_Study.py` & `Sampler.py`**: Automates batch case creation (`cases/case_0`, `cases/case_1`, etc.) by dynamically altering parameters such as inlet flow (InletQ) or bed materials (ManningN).
* **`Measurements.py`**: A validation module for ingesting scalar or vector point measurement data to calculate error norms.

## 🛠️ Quick Start Example (Parametric Study)

Execution control is predominantly handled via clean and scalable JSON files.

```python
from HydroSim.Parametric_Study import Parametric_Study

# Instantiate the controller based on a configuration file
study = Parametric_Study("my_parametric_setup.json")

# Execute the generation of N samples and prepare all SRH-2D directories
samples, parameters = study.create_all_cases()

# Close residual connections, if applicable (e.g., Faceless HEC-RAS)
study.close_and_cleanup()
