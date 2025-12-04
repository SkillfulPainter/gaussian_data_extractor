import os
import re
import csv
from morfeus import Sterimol, read_gjf
import pandas as pd
import statsmodels.api as sm


def extract_data(filepath):
    """
    Extracts Total Energy, HOMO, LUMO, and Dipole Moment from a single Gaussian output file.
    Designed to handle output from optimization tasks, locating the final data at the end of the file.
    Each extraction block is protected independently to ensure a single failure doesn't affect others.
    """
    # Initialize all potential values to None with descriptions
    total_energy = None       # Total SCF Energy
    homo = None               # Highest Occupied Molecular Orbital Energy
    lumo = None               # Lowest Unoccupied Molecular Orbital Energy
    dipole_moment = None      # Dipole Moment (Debye)
    gap = None                # HOMO-LUMO Gap
    eta = None                # Chemical Hardness
    sigma = None              # Chemical Softness
    mu = None                 # Chemical Potential
    chi = None                # Electronegativity
    omega = None              # Electrophilicity Index
    zpe = None                # Zero-point Energy Correction
    e_thermal_corr = None     # Thermal Correction to Energy
    h_corr = None             # Thermal Correction to Enthalpy
    g_corr = None             # Thermal Correction to Gibbs Free Energy
    e_plus_zpe = None         # Sum of electronic and zero-point Energies
    e_plus_e_thermal = None   # Sum of electronic and thermal Energies
    e_plus_h = None           # Sum of electronic and thermal Enthalpies
    e_plus_g = None           # Sum of electronic and thermal Free Energies
    lowest_freq = None        # Lowest Vibrational Frequency
    ke = None                 # Kinetic Energy
    e_n = None                # Electron-Nuclear Attraction Energy
    n_n = None                # Nuclear Repulsion Energy
    mol_mass = None           # Molecular Mass
    cv = None                 # Constant Volume Heat Capacity
    entropy = None            # Entropy

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
            tot_match = re.search(r'Tot=\s*(-?\d+\.\d+)', search_area)
            if tot_match:
                dipole_moment = float(tot_match.group(1))
    except Exception:
        pass

    # --- Extract HOMO and LUMO ---
    try:
        scf_matches = list(re.finditer(r'SCF Done:', content))

        for i in range(len(scf_matches) - 1, -1, -1):
            start_pos = scf_matches[i].start()

            if i + 1 < len(scf_matches):
                end_pos = scf_matches[i + 1].start()
            else:
                end_pos = len(content)

            search_area = content[start_pos:end_pos]

            occ_eigenvalues = re.findall(r'Alpha  occ\. eigenvalues --\s*(.*)', search_area)

            if occ_eigenvalues:
                all_occ_energies = []
                for line in occ_eigenvalues:
                    # Regex extraction handles cases like "-100.0-100.0" perfectly
                    vals = re.findall(r'-?\d+\.\d+', line)
                    all_occ_energies.extend([float(e) for e in vals])
                if all_occ_energies:
                    homo = all_occ_energies[-1]

            virt_eigenvalues = re.findall(r'Alpha virt\. eigenvalues --\s*(.*)', search_area)
            if virt_eigenvalues:
                all_virt_energies = []
                for line in virt_eigenvalues:
                    # Regex extraction for virtual orbitals
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
            # 8. Gap
            gap = lumo - homo

            # 21. Chemical Hardness (Eta)
            eta = gap / 2.0

            # 35. Chemical Potential (Mu)
            mu = (homo + lumo) / 2.0

            # 34. Electronegativity (Chi) (Based on Koopmans' theorem, Chi = -Mu)
            chi = -mu

            if eta != 0:
                # 22. Chemical Softness (Sigma)
                sigma = 1.0 / eta
                # 23. Electrophilicity Index (Omega)
                omega = (mu ** 2) / (2.0 * eta)

    except Exception as e:
        print(f"Error calculating descriptors for {filepath}: {e}")
        pass

    # --- Extract KE, E-N, N-N ---
    try:
        # Matches patterns like: N-N= 4.379D+02 E-N=-1.749D+03 KE= 3.791D+02
        # Gaussian uses 'D' for scientific notation
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
            # Match Total line: Total     116.278     32.687     94.388
            # Columns: E (Thermal) KCal/Mol, CV Cal/Mol-Kelvin, S Cal/Mol-Kelvin
            total_match = re.search(r'Total\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)', thermo_area)
            if total_match:
                cv = float(total_match.group(1))
                entropy = float(total_match.group(2))

        # 14. ZPE
        zpe_match = re.findall(r'Zero-point correction=\s*(-?\d+\.\d+)', content)
        if zpe_match:
            zpe = float(zpe_match[-1])

        # Thermal correction to Energy
        e_thermal_corr_match = re.findall(r'Thermal correction to Energy=\s*(-?\d+\.\d+)', content)
        if e_thermal_corr_match:
            e_thermal_corr = float(e_thermal_corr_match[-1])

        # 15. Enthalpy(H) correction
        h_corr_match = re.findall(r'Thermal correction to Enthalpy=\s*(-?\d+\.\d+)', content)
        if h_corr_match:
            h_corr = float(h_corr_match[-1])

        # 16. G correction
        g_corr_match = re.findall(r'Thermal correction to Gibbs Free Energy=\s*(-?\d+\.\d+)', content)
        if g_corr_match:
            g_corr = float(g_corr_match[-1])

        # 17. E + ZPE
        e_plus_zpe_match = re.findall(r'Sum of electronic and zero-point Energies=\s*(-?\d+\.\d+)', content)
        if e_plus_zpe_match:
            e_plus_zpe = float(e_plus_zpe_match[-1])

        # 18. E + E_thermal
        e_plus_e_thermal_match = re.findall(r'Sum of electronic and thermal Energies=\s*(-?\d+\.\d+)', content)
        if e_plus_e_thermal_match:
            e_plus_e_thermal = float(e_plus_e_thermal_match[-1])

        # 19. E+H
        e_plus_h_match = re.findall(r'Sum of electronic and thermal Enthalpies=\s*(-?\d+\.\d+)', content)
        if e_plus_h_match:
            e_plus_h = float(e_plus_h_match[-1])

        # 20. E+G
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
    elements, coordinates = read_gjf(filepath)
    sterimol = Sterimol(elements, coordinates, axis_atoms[0], axis_atoms[1])
    L = sterimol.L_value
    B_1 = sterimol.B_1_value
    B_5 = sterimol.B_5_value
    return {"L": L, "B_1": B_1, "B_5": B_5}


def main():
    """
    Main function to process files, extract data, and generate CSV report.
    Also performs Multivariate Linear Regression.
    """
    data_folder = r"D:\Path\To\Gaussian\Files"
    output_file = 'results.csv'

    # --- Anion Energy ---
    anion_energy = -459.54813049
    # --- DME Solvent Energy ---
    E_DME_SOLVENT = -308.71907112

    all_fitting_data = []

    # Check if data folder exists
    if not os.path.isdir(data_folder):
        print(f"Error: Folder '{data_folder}' does not exist.")
        return
    sterimol_config = load_sterimol_config(data_folder)
    data_map = load_data(data_folder)

    # Find all unique calculation numbers n
    file_numbers = set()
    for filename in os.listdir(data_folder):
        match = re.match(r'(\d+)-(cation|salt)\.log', filename)
        if match:
            file_numbers.add(int(match.group(1)))

    if not file_numbers:
        print(f"No files matching format n-cation.log or n-salt.log found in '{data_folder}'.")
        return

    # Prepare to write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV Headers
        fieldnames = [
            'number',
            'dependent variable',
            'anion', 'cation', 'salt', 'delta-E (au)', 'DME-Min-Complex-E', 'DME-Min-ΔE (au)',
            # SM block
            'SM-HOMO', 'SM-LUMO', 'SM-Gap',
            'SM-Eta (Hardness)', 'SM-Sigma (Softness)',
            'SM-Mu (Potential)', 'SM-Chi (Electronegativity)',
            'SM-Omega (Electrophilicity)',
            'SM-Dipole Moment(deby)',
            'SM-ZPE', 'SM-E_thermal_corr', 'SM-H_corr', 'SM-G_corr',
            'SM-E+ZPE', 'SM-E+E_thermal', 'SM-E+H', 'SM-E+G', 'SM-LowestFreq',
            'SM-KE', 'SM-E-N', 'SM-N-N', 'SM-Mass(amu)', 'SM-CV', 'SM-S',
            'SM-Sterimol-L', 'SM-Sterimol-B_1', 'SM-Sterimol-B_5',

            # CM block
            'CM-HOMO', 'CM-LUMO', 'CM-Gap',
            'CM-Eta (Hardness)', 'CM-Sigma (Softness)',
            'CM-Mu (Potential)', 'CM-Chi (Electronegativity)',
            'CM-Omega (Electrophilicity)',
            'CM-Dipole Moment(deby)',
            'CM-ZPE', 'CM-E_thermal_corr', 'CM-H_corr', 'CM-G_corr',
            'CM-E+ZPE', 'CM-E+E_thermal', 'CM-E+H', 'CM-E+G', 'CM-LowestFreq',
            'CM-KE', 'CM-E-N', 'CM-N-N', 'CM-Mass(amu)', 'CM-CV', 'CM-S',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Starting file processing, found {len(file_numbers)} groups...")

        # Process each number
        for n in sorted(list(file_numbers)):
            print(f"Processing group {n}...")
            cation_filepath = os.path.join(data_folder, f'{n}-cation.log')
            salt_filepath = os.path.join(data_folder, f'{n}-salt.log')
            cation_gjf_filepath = os.path.join(data_folder, f'{n}-cation.gjf')

            cation_data = extract_data(cation_filepath)
            salt_data = extract_data(salt_filepath)
            axis_atoms = sterimol_config.get(n)
            try:
                if axis_atoms:
                    cation_sterimol = calculate_sterimol(cation_gjf_filepath, axis_atoms)
            except Exception as e:
                print(e)
                cation_sterimol = {"L": None, "B_1": None, "B_5": None}

            # If any file read fails, skip this group
            if not cation_data or not salt_data:
                print(f"Skipping group {n} due to missing files or read errors.")
                continue

            delta_e_au = None
            if cation_data['total_energy'] is not None and salt_data['total_energy'] is not None:
                delta_e_au = salt_data['total_energy'] - cation_data['total_energy'] - anion_energy

            dme_results = {}  # Temporary storage for DME data

            # 1. Find all DME files corresponding to current number n
            dme_files = []
            for filename in os.listdir(data_folder):
                # Match filename format: n-DME-M(index)....log
                match = re.match(rf'{n}-DME-M(\d+)\.\d+\.node1\.log', filename)
                if match:
                    m_index = int(match.group(1))
                    dme_files.append((m_index, filename))

            # Get dependent_variable
            current_cation_energy = cation_data['total_energy']
            dependent_variable = data_map.get(n, None)
            if current_cond is None:
                print(f"Number {n} missing conductivity data, skipping sample.")
                continue

            min_dme_e = None  # Record minimum total energy
            min_dme_delta_au = None  # Record corresponding binding energy

            if dme_files and current_cation_energy is not None:
                for dme_filename in dme_files:
                    dme_filepath = os.path.join(data_folder, dme_filename[1])

                    # Extract data
                    dme_data_extracted = extract_data(dme_filepath)

                    if dme_data_extracted and dme_data_extracted['total_energy'] is not None:
                        e_total = dme_data_extracted['total_energy']

                        # Compare to find minimum
                        if min_dme_e is None or e_total < min_dme_e:
                            min_dme_e = e_total

                # Calculate binding energy after finding minimum
                if min_dme_e is not None:
                    # Formula: ΔE = E_complex_min - E_solvent - E_cation
                    min_dme_delta_au = min_dme_e - E_DME_SOLVENT - current_cation_energy

            # 3. Store in dictionary
            dme_results['DME-Min-Complex-E'] = min_dme_e
            dme_results['DME-Min-ΔE (au)'] = min_dme_delta_au

            # --- Organize Data Row ---
            row_data = {
                'number': n,
                'dependent variable': dependent_variable,
                'anion': anion_energy,
                'cation': cation_data['total_energy'],
                'salt': salt_data['total_energy'],
                'delta-E (au)': delta_e_au,

                # --- SM data fill ---
                'SM-HOMO': cation_data['homo'],
                'SM-LUMO': cation_data['lumo'],
                'SM-Gap': cation_data['gap'],
                'SM-Eta (Hardness)': cation_data['eta'],
                'SM-Sigma (Softness)': cation_data['sigma'],
                'SM-Mu (Potential)': cation_data['mu'],
                'SM-Chi (Electronegativity)': cation_data['chi'],
                'SM-Omega (Electrophilicity)': cation_data['omega'],
                'SM-Dipole Moment(deby)': cation_data['dipole_moment'],
                'SM-ZPE': cation_data['zpe'],
                'SM-E_thermal_corr': cation_data['e_thermal_corr'],
                'SM-H_corr': cation_data['h_corr'],
                'SM-G_corr': cation_data['g_corr'],
                'SM-E+ZPE': cation_data['e_plus_zpe'],
                'SM-E+E_thermal': cation_data['e_plus_e_thermal'],
                'SM-E+H': cation_data['e_plus_h'],
                'SM-E+G': cation_data['e_plus_g'],
                'SM-LowestFreq': cation_data['LowestFreq'],
                'SM-KE': cation_data['KE'],
                'SM-E-N': cation_data['E-N'],
                'SM-N-N': cation_data['N-N'],
                'SM-Mass(amu)': cation_data['MolMass'],
                'SM-CV': cation_data['CV'],
                'SM-S': cation_data['S'],
                'SM-Sterimol-L': cation_sterimol["L"],
                'SM-Sterimol-B_1': cation_sterimol["B_1"],
                'SM-Sterimol-B_5': cation_sterimol["B_5"],

                # --- CM data fill ---
                'CM-HOMO': salt_data['homo'],
                'CM-LUMO': salt_data['lumo'],
                'CM-Gap': salt_data['gap'],
                'CM-Eta (Hardness)': salt_data['eta'],
                'CM-Sigma (Softness)': salt_data['sigma'],
                'CM-Mu (Potential)': salt_data['mu'],
                'CM-Chi (Electronegativity)': salt_data['chi'],
                'CM-Omega (Electrophilicity)': salt_data['omega'],
                'CM-Dipole Moment(deby)': salt_data['dipole_moment'],
                'CM-ZPE': salt_data['zpe'],
                'CM-E_thermal_corr': salt_data['e_thermal_corr'],
                'CM-H_corr': salt_data['h_corr'],
                'CM-G_corr': salt_data['g_corr'],
                'CM-E+ZPE': salt_data['e_plus_zpe'],
                'CM-E+E_thermal': salt_data['e_plus_e_thermal'],
                'CM-E+H': salt_data['e_plus_h'],
                'CM-E+G': salt_data['e_plus_g'],
                'CM-LowestFreq': salt_data['LowestFreq'],
                'CM-KE': salt_data['KE'],
                'CM-E-N': salt_data['E-N'],
                'CM-N-N': salt_data['N-N'],
                'CM-Mass(amu)': salt_data['MolMass'],
                'CM-CV': salt_data['CV'],
                'CM-S': salt_data['S'],
                **dme_results
            }

            # Write to CSV
            writer.writerow(row_data)
            all_fitting_data.append(row_data)

    # --- Multivariate Linear Fitting ---
    print("\n" + "=" * 30)
    print("Starting Multivariate Linear Fitting (Mode: backward)...")

    if not all_fitting_data:
        print("No data available for fitting.")
        return

    # Convert to DataFrame
    df_fit = pd.DataFrame(all_fitting_data)
    df_fit.to_csv(output_file, index=False, encoding='utf-8-sig')

    y_col = '30度电导率'
    if y_col not in df_fit.columns:
        print(f"Error: Target column '{y_col}' not found.")
        return
    y = df_fit[y_col]

    if 'cation' in df_fit.columns:
        start_idx = df_fit.columns.get_loc('cation')
        descriptor_cols = df_fit.columns[start_idx:]
        X = df_fit[descriptor_cols]

        # 1. Data Cleaning
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.dropna(axis=1, how='any')  # Remove columns with NaN values
        X = X.loc[:, (X != X.iloc[0]).any()]  # Remove constant columns

        # Standardization - Important for regression to avoid scale bias
        # Using simple (x - mean) / std
        X_scaled = (X - X.mean()) / X.std()

        print(f"Sample count: {len(y)}, Available descriptors: {X.shape[1]}")

        significance_level = 0.03
        max_init = len(y) - 2

        # Initial Screening if variables exceed sample size
        if X.shape[1] > max_init:
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
            # drop('const') to avoid removing the intercept
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
            # Sort by P-value ascending (smaller P-value is better)
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

            # Find the "worst" variable (largest P-value)
            if contribution_df.empty:
                break

            worst_row = contribution_df.iloc[-1]  # Last row has max P-value
            worst_feature = worst_row['Feature']
            worst_p = worst_row['P-value (Significance)']

            # Decision: Remove if worst P-value > threshold
            if worst_p > significance_level:
                print(f"Decision: Removing descriptor '{worst_feature}' (Low contribution, P={worst_p:.4f} > {significance_level})")
                curr_cols.remove(worst_feature)
                step += 1
            else:
                print(f"\n[Fitting Complete] All remaining descriptors contribute significantly (P < {significance_level}).")
                print("=" * 30)
                print("Final Model Summary:")
                print(model.summary())

                # Save final report
                report_file = 'fitting_report.txt'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(model.summary().as_text())
                print(f"Final fitting report saved to: {report_file}")
                break

    else:
        print("Column 'cation' not found, unable to determine descriptor range.")

    print("=" * 30 + "\n")
    print("Processing complete!")


if __name__ == '__main__':
    main()