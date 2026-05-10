# Reproducibility Guide

This README explains how to reproduce the main results from the paper and how to use the NovaClone algorithm.

## Overview

This repository contains two notebooks:

- **`notebooks/Example.ipynb`**  
  A step-by-step example showing how to use the NovaClone algorithm.  
  Includes explanations of inputs, parameters, and outputs.

- **`notebooks/macsGESTALT_data/clone_calling_macsGESTALT_fig2Res.ipynb`**  
  Reproduces the results presented in the manuscript in section 'Comparison to published experimental data' (figures and analyses).  
  Uses the same data and setup as described in the paper.

Together, these notebooks cover both:
- how to use the method (**Example.ipynb**)  
- and how to reproduce the paper results (**clone_calling_macsGESTALT_fig2Res.ipynb**)

---

## Setup

Clone the repository and install dependencies:

```bash
git clone git@github.com:sprillo/clone-calling.git
cd clone-calling

# Option 1 (recommended)
conda env create -f environment.yml
conda activate nova_clone-env

# Option 2 (pip)
pip install -r requirements.txt

```
## Data

All required data is downloaded within the notebooks using `wget`.
No manual download is needed.

## Running the notebooks

Run the notebooks in the following order:

(Optional) Example notebook – to understand how the algorithm works:
`jupyter notebook notebooks/Example.ipynb`
Reproducibility notebook – to reproduce the results in the paper:
`jupyter notebook notebooks/macsGESTALT_data/clone_calling_macsGESTALT_fig2Res.ipynb`

Run all cells from top to bottom.

## Output
- The outputs from `clone_calling_macsGESTALT_fig2Res.ipynb` correspond to the results shown in the manuscript.
- Figures and intermediate outputs are generated within the notebook (and optionally saved to disk if specified).

## Notes
- Runtime may vary depending on system resources.
- The analysis was tested using the provided environment.
- Some steps may take longer to run.
