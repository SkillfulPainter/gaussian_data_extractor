import os
import re
import csv
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt  # Added missing import
from morfeus import Sterimol, read_gjf


def extract_data(filepath):
    """
    Extracts Total Energy, HOMO, LUMO, and Dipole Moment from a single Gaussian output file.
    Designed to handle output from optimization tasks, locating the final data at the end of the file.
    Each extraction block is protected independently to ensure a single failure doesn't affect others.
    """
    # Initialize all potential values to None with descriptions
    total_energy = None  # Total SCF Energy
    homo = None  # Highest Occupied Molecular Orbital Energy
    lumo = None  # Lowest Unoccupied Molecular Orbital Energy
    dipole_moment = None  # Dipole Moment (Debye)
    gap = None  # HOMO-LUMO Gap
    eta = None  # Chemical Hardness
    sigma = None  # Chemical Softness
    mu = None  # Chemical Potential
    chi = None  # Electronegativity
    omega = None  # Electrophilicity Index
    zpe = None  # Zero-point Energy Correction
    e_thermal_corr = None  # Thermal Correction to Energy
    h_corr = None  # Thermal Correction to Enthalpy
    g_corr = None  # Thermal Correction to Gibbs Free Energy
    e_plus_zpe = None  # Sum of electronic and zero-point Energies
    e_plus_e_thermal = None  # Sum of electronic and thermal Energies
    e_plus_h = None  # Sum of electronic and thermal Enthalpies
    e_plus_g = None  # Sum of electronic and thermal Free Energies
    lowest_freq = None  # Lowest Vibrational Frequency
    ke = None  # Kinetic Energy
    e_n = None  # Electron-Nuclear Attraction Energy
    n_n = None  # Nuclear Repulsion Energy
    mol_mass = None  # Molecular Mass
    cv = None  # Constant Volume Heat Capacity
    entropy = None  # Entropy

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: File not found {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

    # --- Extract Total Energy (SCF Done) ---
    try:
        # Regex matches: SCF Done:  E(RB3LYP) =  -1234.56789012
        scf_matches = re.findall(r'SCF Done:.*?=\s*(-?\d+\.\d+)', content)
        if scf_matches:
            total_energy = float(scf_matches[-1])
    except Exception:
        pass

    # --- Extract Dipole Moment ---
    try:
        dipole_section_start = content.rfind('Dipole moment (field-independent basis, Debye):')
        if dipole_section_start != -1:
            search_area = content[dipole_section_start:]
            # Regex matches: Tot=    1.2345
            tot_match = re.search(r'Tot=\s*(-?\d+\.\d+)', search_area)
            if tot_match:
                dipole_moment = float(tot_match.group(1))
    except Exception:
        pass

    # --- Extract HOMO and LUMO ---
    try:
        # Regex matches: SCF Done:
        scf_matches = list(re.finditer(r'SCF Done:', content))

        for i in range(len(scf_matches) - 1, -1, -1):
            start_pos = scf_matches[i].start()

            if i + 1 < len(scf_matches):
                end_pos = scf_matches[i + 1].start()
            else:
                end_pos = len(content)

            search_area = content[start_pos:end_pos]

            # Regex matches: Alpha  occ. eigenvalues --  -10.12345  -0.67890
            occ_eigenvalues = re.findall(r'Alpha  occ\. eigenvalues --\s*(.*)', search_area)

            if occ_eigenvalues:
                all_occ_energies = []
                for line in occ_eigenvalues:
                    # Regex matches individual numbers: -10.12345 or 0.67890
                    vals = re.findall(r'-?\d+\.\d+', line)
                    all_occ_energies.extend([float(e) for e in vals])
                if all_occ_energies:
                    homo = all_occ_energies[-1]

            # Regex matches: Alpha virt. eigenvalues --   0.12345   1.67890
            virt_eigenvalues = re.findall(r'Alpha virt\. eigenvalues --\s*(.*)', search_area)
            if virt_eigenvalues:
                all_virt_energies = []
                for line in virt_eigenvalues:
                    # Regex matches individual numbers: 0.12345 or 1.67890
                    vals = re.findall(r'-?\d+\.\d+', line)
                    all_virt_energies.extend([float(e) for e in vals])
                if all_virt_energies:
                    lumo = all_virt_energies[0]

            if homo is not None and lumo is not None:
                break
    except Exception as e:
        print(f"Error extracting HOMO/LUMO for {filepath}: {e}")
        pass

    # --- Calculate Descriptors ---
    # (Gap, Hardness, Softness, Chemical Potential, Electronegativity, Electrophilicity Index)
    try:
        if homo is not None and lumo is not None:
            # Gap
            gap = lumo - homo
            # Chemical Hardness (Eta)
            eta = gap / 2.0
            # Chemical Potential (Mu)
            mu = (homo + lumo) / 2.0
            # Electronegativity (Chi) (Based on Koopmans' theorem, Chi = -Mu)
            chi = -mu

            if eta != 0:
                # Chemical Softness (Sigma)
                sigma = 1.0 / eta
                # Electrophilicity Index (Omega)
                omega = (mu ** 2) / (2.0 * eta)

    except Exception as e:
        print(f"Error calculating descriptors for {filepath}: {e}")
        pass

    # --- Extract KE, E-N, N-N ---
    try:
        # Gaussian uses 'D' for scientific notation
        # Regex matches: N-N= 4.379D+02 E-N=-1.749D+03 KE= 3.791D+02
        energy_comp_matches = re.findall(r'N-N=\s*([\d.D+-]+)\s+E-N=\s*([\d.D+-]+)\s+KE=\s*([\d.D+-]+)', content)
        if energy_comp_matches:
            last_match = energy_comp_matches[-1]
            # Replace 'D' with 'E' for Python float parsing
            n_n = float(last_match[0].replace('D', 'E'))
            e_n = float(last_match[1].replace('D', 'E'))
            ke = float(last_match[2].replace('D', 'E'))
    except Exception as e:
        print(f"Error extracting energy components for {filepath}: {e}")
        pass

    # --- Extract Molecular Mass ---
    try:
        # Regex matches: Molecular mass:   180.12345 amu
        mass_match = re.search(r'Molecular mass:\s*([\d.]+)\s+amu', content)
        if mass_match:
            mol_mass = float(mass_match.group(1))
    except Exception as e:
        print(f"Error extracting molecular mass for {filepath}: {e}")
        pass

    # --- Extract Frequencies ---
    try:
        # Locate the last occurrence of "Harmonic frequencies"
        freq_start_idx = content.rfind("Harmonic frequencies")

        search_area = content
        if freq_start_idx != -1:
            search_area = content[freq_start_idx:]

        # Regex matches: Frequencies --    100.1234    200.5678    300.9101
        freq_matches = re.findall(r'Frequencies --\s+(.*)', search_area)

        if freq_matches:
            all_freqs = []
            for line in freq_matches:
                try:
                    vals = [float(v) for v in line.split()]
                    all_freqs.extend(vals)
                except ValueError:
                    continue
            if all_freqs:
                # Gaussian frequencies are sorted algebraically.
                # The first one is the lowest frequency (or largest negative if imaginary).
                lowest_freq = all_freqs[0]

    except Exception as e:
        print(f"Error extracting frequencies for {filepath}: {e}")
        pass

    # --- Extract Thermochemistry Data ---
    try:
        # CV and S are located in the Thermochemistry section
        thermo_start = content.rfind("- Thermochemistry -")
        if thermo_start != -1:
            thermo_area = content[thermo_start:]
            # Columns: E (Thermal) KCal/Mol, CV Cal/Mol-Kelvin, S Cal/Mol-Kelvin
            # Regex matches: Total     116.278     32.687     94.388
            total_match = re.search(r'Total\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)', thermo_area)
            if total_match:
                cv = float(total_match.group(1))
                entropy = float(total_match.group(2))

        # ZPE
        # Regex matches: Zero-point correction=                           0.123456
        zpe_match = re.findall(r'Zero-point correction=\s*(-?\d+\.\d+)', content)
        if zpe_match:
            zpe = float(zpe_match[-1])

        # Thermal correction to Energy
        # Regex matches: Thermal correction to Energy=                    0.134567
        e_thermal_corr_match = re.findall(r'Thermal correction to Energy=\s*(-?\d+\.\d+)', content)
        if e_thermal_corr_match:
            e_thermal_corr = float(e_thermal_corr_match[-1])

        # Enthalpy(H) correction
        # Regex matches: Thermal correction to Enthalpy=                  0.135511
        h_corr_match = re.findall(r'Thermal correction to Enthalpy=\s*(-?\d+\.\d+)', content)
        if h_corr_match:
            h_corr = float(h_corr_match[-1])

        # G correction
        # Regex matches: Thermal correction to Gibbs Free Energy=         0.098765
        g_corr_match = re.findall(r'Thermal correction to Gibbs Free Energy=\s*(-?\d+\.\d+)', content)
        if g_corr_match:
            g_corr = float(g_corr_match[-1])

        # E + ZPE
        # Regex matches: Sum of electronic and zero-point Energies=           -1234.123456
        e_plus_zpe_match = re.findall(r'Sum of electronic and zero-point Energies=\s*(-?\d+\.\d+)', content)
        if e_plus_zpe_match:
            e_plus_zpe = float(e_plus_zpe_match[-1])

        # E + E_thermal
        # Regex matches: Sum of electronic and thermal Energies=              -1234.112345
        e_plus_e_thermal_match = re.findall(r'Sum of electronic and thermal Energies=\s*(-?\d+\.\d+)', content)
        if e_plus_e_thermal_match:
            e_plus_e_thermal = float(e_plus_e_thermal_match[-1])

        # E+H
        # Regex matches: Sum of electronic and thermal Enthalpies=            -1234.111401
        e_plus_h_match = re.findall(r'Sum of electronic and thermal Enthalpies=\s*(-?\d+\.\d+)', content)
        if e_plus_h_match:
            e_plus_h = float(e_plus_h_match[-1])

        # E+G
        # Regex matches: Sum of electronic and thermal Free Energies=         -1234.148147
        e_plus_g_match = re.findall(r'Sum of electronic and thermal Free Energies=\s*(-?\d+\.\d+)', content)
        if e_plus_g_match:
            e_plus_g = float(e_plus_g_match[-1])

    except Exception as e:
        print(f"Error extracting thermochemistry data for {filepath}: {e}")
        pass

    return {
        'total_energy': total_energy,
        'homo': homo,
        'lumo': lumo,
        'dipole_moment': dipole_moment,
        'gap': gap,
        'eta': eta,
        'sigma': sigma,
        'mu': mu,
        'chi': chi,
        'omega': omega,
        'zpe': zpe,
        'e_thermal_corr': e_thermal_corr,
        'h_corr': h_corr,
        'g_corr': g_corr,
        'e_plus_zpe': e_plus_zpe,
        'e_plus_e_thermal': e_plus_e_thermal,
        'e_plus_h': e_plus_h,
        'e_plus_g': e_plus_g,
        'LowestFreq': lowest_freq,
        'KE': ke,
        'E-N': e_n,
        'N-N': n_n,
        'MolMass': mol_mass,
        'CV': cv,
        'S': entropy,
    }


def load_sterimol_config(data_folder, excel_filename="data.xlsx"):
    """
    Reads the Excel file in the specified directory and parses Sterimol parameters.
    Returns a dictionary: {n: (atom1, atom2)}
    """
    config = {}
    filepath = os.path.join(data_folder, excel_filename)

    if not os.path.exists(filepath):
        print(f"Note: File not found {filepath}")
    else:
        print(f"Successfully located data file: {filepath}")

    try:
        df = pd.read_excel(filepath)
        # Check if columns exist
        if 'number' not in df.columns or 'sterimol axis atoms' not in df.columns:
            print(f"Warning: Config file {filepath} missing columns 'number' or 'sterimol axis atoms'.")
            print(f"Available columns: {df.columns.tolist()}")
            return config

        for index, row in df.iterrows():
            try:
                # Extract n value
                n_val = row['number']
                if pd.isna(n_val):
                    continue
                n = int(n_val)

                # Extract parameter string, e.g., "6,1"
                params_str = str(row['sterimol axis atoms'])
                if pd.isna(params_str):
                    continue

                # Handle full-width comma and half-width comma
                params_str = params_str.replace('，', ',')
                parts = params_str.split(',')

                if len(parts) >= 2:
                    atom1 = int(float(parts[0]))
                    atom2 = int(float(parts[1]))
                    config[n] = (atom1, atom2)
            except Exception as e:
                print(f"Error parsing sterimol file row {index + 1}: {e}")
                continue
        print(f"Successfully loaded {len(config)} Sterimol parameter configurations.")

    except Exception as e:
        print(f"Critical error reading config file: {e}")

    return config


def load_data(data_folder, excel_filename="data.xlsx"):
    """
    Reads conductivity data, returns dictionary {number: conductivity}
    """
    cond_map = {}
    filepath = os.path.join(data_folder, excel_filename)
    if not os.path.exists(filepath):
        print(f"Warning: data file not found {filepath}")
        return cond_map

    try:
        df = pd.read_excel(filepath)
        # Ensure correct column names
        if 'number' not in df.columns or 'dependent variable' not in df.columns:
            print(f"Warning: {excel_filename} missing columns 'number' or 'dependent variable'")
            return cond_map

        for _, row in df.iterrows():
            n_val = row['number']
            cond_val = row['dependent variable']
            if pd.notna(n_val) and pd.notna(cond_val):
                cond_map[int(n_val)] = float(cond_val)
        print(f"Successfully loaded {len(cond_map)} data entries.")
    except Exception as e:
        print(f"Error reading data file: {e}")

    return cond_map


def calculate_sterimol(filepath, axis_atoms=(1, 6)):
    """
    Calculates Sterimol parameters using the morfeus library.
    Requires a .gjf file with geometries.
    """
    try:
        elements, coordinates = read_gjf(filepath)
        sterimol = Sterimol(elements, coordinates, axis_atoms[0], axis_atoms[1])
        L = sterimol.L_value
        B_1 = sterimol.B_1_value
        B_5 = sterimol.B_5_value
        return {"L": L, "B_1": B_1, "B_5": B_5}
    except Exception as e:
        print(f"Sterimol Calculation Failed for {filepath}: {e}")
        return {"L": None, "B_1": None, "B_5": None}


def main():
    """
    Main function to process files, extract data, and generate CSV report.
    """
    data_folder = r"D:\Research\GaussianData"
    output_file = 'results.csv'

    all_fitting_data = []

    # Check if data folder exists
    if not os.path.isdir(data_folder):
        print(f"Error: Folder '{data_folder}' does not exist.")
        return

    # Load configurations
    sterimol_config = load_sterimol_config(data_folder)
    data_map = load_data(data_folder)

    # Prepare to write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'number',
            'filename',
            'dependent variable',
            'Total Energy',
            'HOMO', 'LUMO', 'Gap',
            'Eta (Hardness)', 'Sigma (Softness)',
            'Mu (Potential)', 'Chi (Electronegativity)',
            'Omega (Electrophilicity)',
            'Dipole Moment(debye)',
            'ZPE', 'E_thermal_corr', 'H_corr', 'G_corr',
            'E+ZPE', 'E+E_thermal', 'E+H', 'E+G', 'LowestFreq',
            'KE', 'E-N', 'N-N', 'Mass(amu)', 'CV', 'S',
            'Sterimol-L', 'Sterimol-B_1', 'Sterimol-B_5'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print("Starting general file processing...")

        # Scan folder for all log files matching pattern "n-something.log"
        files_found = 0
        for filename in os.listdir(data_folder):
            # Regex matches: 1-*.log (Group 1: 1, Group 2: cation)
            match = re.match(r'^(\d+)-(.+)\.log$', filename)
            if not match:
                continue

            n = int(match.group(1))
            filepath = os.path.join(data_folder, filename)

            print(f"Processing file: {filename} (Number: {n})")

            # Extract Quantum Data
            qm_data = extract_data(filepath)

            if not qm_data or qm_data['total_energy'] is None:
                print(f"Skipping {filename} due to extraction failure.")
                continue

            # Handle Sterimol (requires .gjf file with same name)
            gjf_filename = filename.replace('.log', '.gjf')
            gjf_filepath = os.path.join(data_folder, gjf_filename)

            sterimol_data = {"L": None, "B_1": None, "B_5": None}
            axis_atoms = sterimol_config.get(n)

            # Only calculate sterimol if config exists and .gjf exists
            if axis_atoms and os.path.exists(gjf_filepath):
                sterimol_data = calculate_sterimol(gjf_filepath, axis_atoms)

            # Get dependent variable
            dependent_variable = data_map.get(n, None)

            # --- Organize Data Row ---
            row_data = {
                'number': n,
                'filename': filename,
                'dependent variable': dependent_variable,
                'Total Energy': qm_data['total_energy'],
                'HOMO': qm_data['homo'],
                'LUMO': qm_data['lumo'],
                'Gap': qm_data['gap'],
                'Eta (Hardness)': qm_data['eta'],
                'Sigma (Softness)': qm_data['sigma'],
                'Mu (Potential)': qm_data['mu'],
                'Chi (Electronegativity)': qm_data['chi'],
                'Omega (Electrophilicity)': qm_data['omega'],
                'Dipole Moment(debye)': qm_data['dipole_moment'],
                'ZPE': qm_data['zpe'],
                'E_thermal_corr': qm_data['e_thermal_corr'],
                'H_corr': qm_data['h_corr'],
                'G_corr': qm_data['g_corr'],
                'E+ZPE': qm_data['e_plus_zpe'],
                'E+E_thermal': qm_data['e_plus_e_thermal'],
                'E+H': qm_data['e_plus_h'],
                'E+G': qm_data['e_plus_g'],
                'LowestFreq': qm_data['LowestFreq'],
                'KE': qm_data['KE'],
                'E-N': qm_data['E-N'],
                'N-N': qm_data['N-N'],
                'Mass(amu)': qm_data['MolMass'],
                'CV': qm_data['CV'],
                'S': qm_data['S'],
                'Sterimol-L': sterimol_data["L"],
                'Sterimol-B_1': sterimol_data["B_1"],
                'Sterimol-B_5': sterimol_data["B_5"],
            }

            writer.writerow(row_data)

            # Only add to fitting data if dependent variable exists
            if dependent_variable is not None:
                all_fitting_data.append(row_data)

            files_found += 1

        print(f"Processing complete. Processed {files_found} files.")

    # --- Multivariate Linear Fitting ---
    print("\n" + "=" * 30)
    print("Starting Multivariate Linear Fitting (Mode: backward)...")

    if not all_fitting_data:
        print("No data available for fitting (check if 'dependent variable' is present).")
        return

    # Convert to DataFrame
    df_fit = pd.DataFrame(all_fitting_data)

    # Target column
    y_col = 'dependent variable'

    if y_col not in df_fit.columns:
        print(f"Error: Target column '{y_col}' not found.")
        return

    # Drop filename and number for regression purposes
    df_fit_numeric = df_fit.drop(columns=['filename', 'number'], errors='ignore')

    # Handle missing values in dependent variable if any slipped through
    df_fit_numeric = df_fit_numeric.dropna(subset=[y_col])

    y = df_fit_numeric[y_col]
    X = df_fit_numeric.drop(columns=[y_col])

    # 1. Data Cleaning
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='any')  # Remove columns with NaN values
    X = X.loc[:, (X != X.iloc[0]).any()]  # Remove constant columns

    if X.empty:
        print("No valid descriptor columns remaining after cleaning.")
        return

    # Standardization
    X_scaled = (X - X.mean()) / X.std()

    print(f"Sample count: {len(y)}, Available descriptors: {X.shape[1]}")

    significance_level = 0.05
    max_init = len(y) - 2

    # Initial Screening if variables exceed sample size
    if X.shape[1] > max_init and max_init > 0:
        print(f"Too many variables, performing initial screening, keeping top {max_init}...")
        corrs = X_scaled.apply(lambda x: x.corr(y, method='spearman')).abs().sort_values(ascending=False)
        X_curr = X_scaled[corrs.index[:max_init]]
    else:
        X_curr = X_scaled.copy()

    curr_cols = X_curr.columns.tolist()
    step = 1

    while True:
        if not curr_cols:
            print("All descriptors removed, unable to build model.")
            break

        # Prepare data and add constant (intercept)
        X_curr = X[curr_cols]
        X_with_const = sm.add_constant(X_curr)

        try:
            # Fit model
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            print(f"Fitting error: {e}")
            break

        # Get statistics
        p_values = model.pvalues.drop('const', errors='ignore')
        t_values = model.tvalues.drop('const', errors='ignore')
        coeffs = model.params.drop('const', errors='ignore')

        # Create Contribution Table
        contribution_df = pd.DataFrame({
            'Feature': p_values.index,
            'Coeff (Weight)': coeffs.values,
            't-value (Strength)': t_values.values,
            'P-value (Significance)': p_values.values
        })
        contribution_df.sort_values('P-value (Significance)', ascending=True, inplace=True)

        print(f"\n--- Round {step} Fitting ---")
        print(f"Current R-squared: {model.rsquared:.4f}")
        print("Descriptor Contribution (Sorted by Significance, smaller P is better):")
        # Format output
        print(contribution_df.to_string(index=False, formatters={
            'Coeff (Weight)': '{:.4e}'.format,
            't-value (Strength)': '{:.4f}'.format,
            'P-value (Significance)': '{:.4f}'.format
        }))
        # Find the "worst" variable
        if contribution_df.empty:
            break

        worst_row = contribution_df.iloc[-1]
        worst_feature = worst_row['Feature']
        worst_p = worst_row['P-value (Significance)']

        # Decision
        if worst_p > significance_level:
            print(f"Decision: Removing '{worst_feature}' (P={worst_p:.4f} > {significance_level})")
            curr_cols.remove(worst_feature)
            step += 1
        else:
            print(f"\n[Fitting Complete] All descriptors significant (P < {significance_level}).")
            print("=" * 30)
            print("Final Model Summary:")
            print(model.summary())

            # Save reports
            with open('fitting_report.txt', 'w', encoding='utf-8') as f:
                f.write(model.summary().as_text())

            # Plotting
            try:
                print("Generating prediction plot...")
                y_pred = model.predict(X_with_const)
                plt.figure(figsize=(6, 6))
                plt.scatter(y, y_pred, color='blue', alpha=0.6, edgecolors='k', label='Data Points')

                combined_min = min(y.min(), y_pred.min())
                combined_max = max(y.max(), y_pred.max())
                margin = (combined_max - combined_min) * 0.05
                plot_limit = [combined_min - margin, combined_max + margin]

                plt.plot(plot_limit, plot_limit, color='red', linestyle='--', linewidth=2, label='Perfect Fit')
                plt.xlim(plot_limit)
                plt.ylim(plot_limit)
                plt.xlabel('Experimental', fontsize=12)
                plt.ylabel('Predicted', fontsize=12)
                plt.title(f'Experimental vs Predicted\n$R^2 = {model.rsquared:.4f}$', fontsize=14)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
                print("Plot saved.")
            except Exception as e:
                print(f"Error plotting: {e}")
            break

    print("=" * 30 + "\n")
    print("Processing complete!")


if __name__ == '__main__':
    main()