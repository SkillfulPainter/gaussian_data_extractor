# Gaussian Output Data Extractor & QSAR Modeling Tool

## Overview
This Python tool is designed to automate the extraction of quantum chemical descriptors from Gaussian calculation output files (`.log`) and perform subsequent multivariate statistical analysis. 

It is specifically tailored for analyzing DFT calculations of small organic molecules, extracting physical organic descriptors, and performing stepwise linear regression to model experimental properties.
* The current version only contains the data extractor part. The QSAR modeling part will be uploaded in the future.

## Key Features
* **Batch Extraction:** Automatically processes multiple molecules based on numerical indices found in filenames.
* **Robust Parsing:** Extracts SCF energies, orbital energies (HOMO/LUMO), Dipole moments, and Frequency data using regex.
* **Descriptor Calculation:** Computes DFT-based reactivity indices (Hardness, Softness, Electrophilicity, etc.).
* **Steric Analysis:** Calculates Sterimol parameters ($L, B_1, B_5$) using `morfeus` based on `.gjf` geometries.
* **Complexation Energy:** Automatic calculation of binding energies ($\Delta E$) for salts and solvent-cation complexes.
* **Statistical Modeling:** Performs automatic Backward Stepwise OLS Regression to identify significant descriptors correlated with experimental data.

## Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install pandas statsmodels morfeus-py openpyxl
```
*Note: The script also uses standard libraries: `os`, `re`, `csv`.*

## File Directory & Naming Convention

The script relies on a strict file naming convention to associate files with a specific sample ID (integer `n`).

### 1. Gaussian Output Files
Place all files in a single directory. Replace `n` with the sample number (e.g., `1`, `2`, `10`...):

| File Type           | Naming Pattern | Description |
|:--------------------|:---------------| :--- |
| **Gaussian Output** | `n-**.log`     | Output log for the isolated cation. |

### 2. Configuration & Experimental Data (`data.xlsx`)
You also need to provide an Excel file named **`data.xlsx`** in the same directory. This file serves two purposes: providing the experimental target variable (for regression) and configuration for steric calculations.

#### **Required Data Format**

The Excel file must contain the following columns (headers are case-sensitive):

| Column Name | Description                                                                     | Example |
| :--- |:--------------------------------------------------------------------------------| :--- |
| **`number`** | The Sample ID corresponding to `n` in filenames.                                | `1` |
| **`dependent variable`** | The experimental value (target Y) to predict.                                   | `5.4` |
| **`sterimol axis atoms`** | Atom indices to define an axis for Sterimol calculations, separated by a comma. | `1,6` |

**Example `data.xlsx` content:**

| number | dependent variable | sterimol axis atoms |
| :--- | :--- | :--- |
| 1 | 8.23 | 1,6 |
| 2 | 7.45 | 1,5 |
| 3 | 9.10 | 2,7 |

## Extracted Descriptors

The script extracts and calculates the following descriptors:

### Electronic & Reactivity
* **Energies:** HOMO, LUMO, HOMO-LUMO Gap.
* **DFT Indices:**
    * Chemical Hardness ($\eta$)
    * Chemical Softness ($\sigma$)
    * Chemical Potential ($\mu$)
    * Electronegativity ($\chi$)
    * Electrophilicity Index ($\omega$)
* **Dipole Moment:** Field-independent basis (Debye).

### Thermodynamic & Energetic
* **Energies:** Total SCF Energy, Kinetic Energy (KE), Nuclear Repulsion (N-N), Electron-Nuclear (E-N).
* **Corrections:** ZPE, Thermal Corrections to Energy, Enthalpy (H), and Gibbs Free Energy (G).
* **Thermochemistry:** Entropy ($S$), Heat Capacity ($C_v$).
* **Binding Energies:** $\Delta E$ for Salt formation and Solvent-Cation interaction.

### Structural & Steric
* **Sterimol Parameters:** $L$ (Length), $B_1$ (Min width), $B_5$ (Max width).
* **Frequencies:** Lowest vibrational frequency.
* **Mass:** Molecular mass.

---

## Detailed Usage Guide

Follow these steps to run the analysis:

### Step 1: Prepare Your Directory
Create a folder (e.g., `D:\Research\GaussianData`) and ensure it contains:
1.  All your `.log` and `.gjf` files named correctly (see Naming Convention).
2.  The `data.xlsx` file containing your experimental data and sterimol configs.

### Step 2: Configure the Script
Open `extract_gaussian_data.py` in a text editor or IDE. Locate the `main()` function and update the `data_folder` variable to point to your directory:

```python
def main():
    # ...
    data_folder = r"D:\Research\GaussianData"  # <--- Update this path
    output_file = 'results.csv'
    # ...
```

*Optional:* If your system uses a different Anion or Solvent, update the energy constants at the top of the `main()` function:
```python
    anion_energy = -459.54813049
    E_DME_SOLVENT = -308.71907112
```

### Step 3: Run the Script
Open your terminal or command prompt, navigate to the folder containing the python script, and run:

```bash
python extract_gaussian_data.py
```

### Step 4: Monitor Console Output
The script will provide real-time feedback in the console:

1.  **Loading:** It will confirm that `data.xlsx` was loaded and how many Sterimol configs were found.
2.  **Processing:** It will iterate through every group number found:
    > `Processing group 1...`
3.  **Fitting:** Once extraction is done, it begins the Multivariate Linear Regression (Backward Elimination):
    > `Starting Multivariate Linear Fitting (Mode: backward)...`
    > `--- Round 1 Fitting ---`
    > `Descriptor Contribution...`
    > `Decision: Removing descriptor 'SM-LUMO' (Low contribution, P=0.85...)`

### Step 5: Check Outputs
After execution, two new files will be generated in your working directory:

1.  **`results.csv`**: A comprehensive dataset containing every extracted descriptor for every molecule. This is your raw data for further analysis.
2.  **`fitting_report.txt`**: The final statistical summary of the best regression model found, including R-squared, F-statistic, and coefficients.

## Troubleshooting

* **`Warning: File not found ...`**: The script cannot find a specific log file. Double-check that your files are named exactly `n-cation.log`, `n-salt.log`, etc.
* **`Warning: Config file ... missing columns`**: Your `data.xlsx` headers are likely incorrect. They must exactly match `number`, `dependent variable`, and `sterimol axis atoms`.
* **`Error extracting HOMO/LUMO`**: The script failed to parse the orbital energies. Ensure your Gaussian jobs included orbital printing (standard in optimization jobs) and finished successfully (`SCF Done`).
* **Empty `fitting_report.txt`**: If the regression fails, check if `results.csv` contains `NaN` values (blank cells). The regression tool removes columns containing any missing data.