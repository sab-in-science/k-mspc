# Kernel-MSPC Optimisation Toolbox

This repository contains the Julia code and tools accompanying the submitted paper:

**"Optimising Kernel-based Multivariate Statistical Process Control"**
Zina-Sabrina Duma, Victoria Jorry, Tuomas Sihvonen, Satu-Pia Reinikainen, Lassi Roininen 
> [📄 Read the preprint of the submitted paper](https://arxiv.org/pdf/2505.01556)  
> [🎓 Associated Institution: LUT University](https://www.lut.fi)

## What’s Inside?

This toolbox provides an efficient and interpretable way to optimise **Kernel Multivariate Statistical Process Control (K-MSPC)** using **Kernel Flows**. It enhances fault detection in industrial processes by learning the kernel parameters of a set kernel function ("kernel" = i.e. "gaussian", "matern5/2", ...), or the kernel type and the kernel parameters ("kernelType" = "family").

### Key Features

- **Fault detection via K-PCA, with parameter optimisation via K-PCR**
- **Multiple kernel types**: Gaussian, Matérn (½, 3/2, 5/2), Cauchy, combination of kernels
- **Supports learning parameters (a) for the whole dataset ("combined"), and (b) individually for each variable ("individual" or "individualScale")**
- **Optimisation via Kernel Flows** (gradient-based learning)
- Benchmarked on the **Tennessee Eastman Process**

---

## Installation

Clone the repository and ensure you have [Julia](https://julialang.org/) installed (tested on Julia ≥ 1.8).

```bash
git clone https://github.com/sab-in-science/k-mspc.git
cd k-mspc
```

Ensure the following packages are added:
```julia
using Pkg
Pkg.add("DelimitedFiles")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Distributions")
```
## 🚀 Quick Start

### 1. Prepare your model dictionary

```julia
model = Dict(
    :dataset => "tenessee",
    :dataSubset => [1, 3],
    :testFault => 3,
    :kernelVersion => "individual",  # or "individualScale", "combined", etc.
    :kernelType => "cauchy",         # or "gaussian", "matern3/2", etc.
    :scale => true,
    :dim => 4,
    :α => 0.99,
    :learnRate => 0.01,
    :iter => 200,
    :nsamp => 20,
    :sp => 0.3,
    :gradClip => 1.0,
    :paper => 1
)
```

### 2. Load the data

```julia
include("MSPC.jl")
model = loadData(model)
```

### 3. Run the optimiser

```julia
model, paramHist, lossHist, gradHist = optimize_parameters(model)
```

---

## 📈 Output

- `model[:bestParam]`: Optimal kernel parameters
- `lossHist`: Loss evolution over iterations
- `paramHist`: Parameter values during learning
- T² and SPEx control charts can be built from the optimised model

---

## 📁 Data

The script assumes the Tennessee Eastman datasets are stored under `Fault Detection Data/` as `.dat` files.

Example structure:
```
Fault Detection Data/
├── d00_te.dat
├── d01.dat
├── d01_te.dat
...
```

---

## 🧑‍🔬 Citation

If you use this toolbox, please cite:

```bibtex
@misc{duma2025optimisingkernelbasedmultivariatestatistical,
      title={Optimising Kernel-based Multivariate Statistical Process Control}, 
      author={Zina-Sabrina Duma and Victoria Jorry and Tuomas Sihvonen and Satu-Pia Reinikainen and Lassi Roininen},
      year={2025},
      eprint={2505.01556},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2505.01556}, 
}
```

---

## 📬 Contact

For questions, reach out to [Zina-Sabrina Duma](mailto:Zina-Sabrina.Duma@lut.fi)  
Or open an issue in this repository.
