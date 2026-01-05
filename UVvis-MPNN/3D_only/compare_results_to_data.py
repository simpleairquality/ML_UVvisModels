#!/usr/bin/env python3
"""
Compare predicted UV-Vis spectra with experimental NIST data.

Usage:
    python compare_predicted_vs_nist_spectra.py predicted_spectra.csv -o output_comparison.csv

Input CSV format:
    SMILES,220.0,221.0,222.0,...,400.0
    
Output will include:
    - R² correlation
    - Mean absolute error
    - Root mean square error
    - Spectral overlap metrics
"""

import sys
import csv
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    import requests
    from urllib.parse import quote
except ImportError:
    print("Error: requests not installed. Install with: pip install requests")
    sys.exit(1)

try:
    import nistchempy as nist
except ImportError:
    print("Error: nistchempy not installed. Install with: pip install nistchempy")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    print("Error: RDKit not installed. Install with: pip install rdkit")
    sys.exit(1)


def parse_jcamp_dx(jcamp_text: str, force_log_epsilon: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse JCAMP-DX format UV-Vis spectrum.
    
    Args:
        jcamp_text: JCAMP-DX format text
        force_log_epsilon: Force conversion from log-epsilon even if not detected
    
    Returns:
        wavelengths (nm), absorbance values (converted from log-epsilon to linear scale)
    """
    wavelengths = []
    absorbances = []
    
    lines = jcamp_text.split('\n')
    in_data_section = False
    is_log_epsilon = force_log_epsilon
    
    for line in lines:
        line = line.strip()
        
        # Check if data is in log-epsilon format
        if not force_log_epsilon and ('##YUNITS' in line.upper() or '##YUNIT' in line.upper()):
            if 'LOG' in line.upper() or 'EPSILON' in line.upper():
                is_log_epsilon = True
        
        # Look for data table start - handle both formats
        if line.startswith('##XYDATA=') or line.startswith('##XYPOINTS='):
            in_data_section = True
            # Check if data starts on same line (e.g., ##XYPOINTS=(XY..XY))
            if '(' in line:
                continue  # Data will be on following lines
            continue
        
        # End of data section
        if line.startswith('##'):
            if in_data_section and line.startswith('##END'):
                in_data_section = False
            continue
        
        # Parse data lines
        if in_data_section and line:
            # Remove any trailing comments or parentheses
            line = line.rstrip(')')
            
            # JCAMP format: "wavelength,absorbance" with comma separator
            # Split by comma first, then by whitespace as fallback
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    wl = float(parts[0].strip())
                    abs_val = float(parts[1].strip())
                    wavelengths.append(wl)
                    absorbances.append(abs_val)
                except ValueError:
                    continue
    
    if len(wavelengths) == 0:
        raise ValueError("No data points found in JCAMP-DX file")
    
    wavelengths = np.array(wavelengths)
    absorbances = np.array(absorbances)
    
    # Convert from log-epsilon to linear scale if needed
    # log(epsilon) -> epsilon (molar extinction coefficient)
    # For comparison with normalized spectra, we just need relative values
    if is_log_epsilon:
        print("  Converting from log-epsilon to linear scale")
        absorbances = 10 ** absorbances
    
    return wavelengths, absorbances


def normalize_spectrum(wavelengths: np.ndarray, absorbances: np.ndarray) -> np.ndarray:
    """Normalize spectrum to max value of 1.0 (matches original plot.py behavior)"""
    max_abs = np.max(absorbances)
    if max_abs > 0:
        return absorbances / max_abs
    return absorbances


def interpolate_spectrum(wavelengths: np.ndarray, absorbances: np.ndarray, 
                         target_wavelengths: np.ndarray) -> np.ndarray:
    """
    Interpolate spectrum to match target wavelength grid.
    Handles both ascending and descending wavelength orders.
    """
    # Check if wavelengths are in descending order
    if len(wavelengths) > 1 and wavelengths[0] > wavelengths[-1]:
        # Reverse both arrays to make ascending
        wavelengths = wavelengths[::-1]
        absorbances = absorbances[::-1]
    
    return np.interp(target_wavelengths, wavelengths, absorbances)


def calculate_metrics(predicted: np.ndarray, experimental: np.ndarray) -> Dict[str, float]:
    """Calculate comparison metrics between predicted and experimental spectra"""
    
    # Remove any NaN values
    mask = ~(np.isnan(predicted) | np.isnan(experimental))
    pred = predicted[mask]
    exp = experimental[mask]
    
    if len(pred) == 0:
        return {
            'r_squared': np.nan,
            'pearson_r': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'spectral_angle': np.nan,
            'overlap_coefficient': np.nan
        }
    
    # Pearson correlation coefficient (measures linear relationship, doesn't penalize shifts)
    if len(pred) > 1:
        pearson_r = np.corrcoef(pred, exp)[0, 1]
    else:
        pearson_r = np.nan
    
    # R² (coefficient of determination)
    ss_res = np.sum((exp - pred) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred - exp))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred - exp) ** 2))
    
    # Spectral Angle Mapper (SAM) - measures angular difference
    dot_product = np.dot(pred, exp)
    norm_pred = np.linalg.norm(pred)
    norm_exp = np.linalg.norm(exp)
    if norm_pred > 0 and norm_exp > 0:
        cos_angle = dot_product / (norm_pred * norm_exp)
        cos_angle = np.clip(cos_angle, -1, 1)
        spectral_angle = np.arccos(cos_angle) * 180 / np.pi  # in degrees
    else:
        spectral_angle = np.nan
    
    # Overlap coefficient (Szymkiewicz–Simpson coefficient)
    min_sum = np.sum(np.minimum(pred, exp))
    max_individual = max(np.sum(pred), np.sum(exp))
    overlap_coef = min_sum / max_individual if max_individual > 0 else 0
    
    return {
        'r_squared': r_squared,
        'pearson_r': pearson_r,
        'mae': mae,
        'rmse': rmse,
        'spectral_angle': spectral_angle,
        'overlap_coefficient': overlap_coef
    }


def fetch_nist_spectrum(smiles: str, max_results: int = 5, 
                       force_log_epsilon: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fetch UV-Vis spectrum from NIST Chemistry WebBook for a given SMILES.
    Uses nistchempy to search and retrieve UV-Vis spectra.
    
    Args:
        smiles: SMILES string
        max_results: Maximum number of NIST results to check
        force_log_epsilon: Force log-epsilon conversion
    
    Returns:
        (wavelengths, absorbances) or None if not found
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Warning: Invalid SMILES: {smiles}")
            return None
        
        # Generate InChI and InChIKey
        inchi = Chem.MolToInchi(mol)
        inchi_key = Chem.InchiToInchiKey(inchi)
        
        print(f"  Searching NIST for InChI Key: {inchi_key}")
        
        # Search NIST using nistchempy
        try:
            result = nist.run_search(identifier=inchi, search_type='inchi')
            result.load_found_compounds()
            
            if len(result.compounds) == 0:
                print(f"  No compounds found in NIST")
                return None
            
            if len(result.compounds) > 1:
                print(f"  Warning: Found {len(result.compounds)} compounds, using first match")
            
            # Use the first compound
            compound = result.compounds[0]
            print(f"  Found compound: {compound.name} (ID: {compound.ID})")
            
            # Rate limiting
            time.sleep(1)
            
            # Get UV-Vis spectra
            compound.get_uv_spectra()
            
            if not hasattr(compound, 'uv_specs') or len(compound.uv_specs) == 0:
                print(f"  No UV-Vis spectra available for this compound")
                return None
            
            print(f"  Found {len(compound.uv_specs)} UV-Vis spectrum/spectra")
            
            # Use the first UV-Vis spectrum
            spec = compound.uv_specs[0]
            
            if not hasattr(spec, 'jdx_text') or not spec.jdx_text:
                print(f"  UV-Vis spectrum has no JCAMP-DX data")
                return None
            
            # Parse JCAMP-DX data
            wavelengths, absorbances = parse_jcamp_dx(spec.jdx_text, force_log_epsilon)
            
            print(f"  Successfully parsed spectrum with {len(wavelengths)} data points")
            
            return wavelengths, absorbances
            
        except AttributeError as e:
            print(f"  Error accessing compound data: {e}")
            return None
        except Exception as e:
            print(f"  Error during NIST search: {e}")
            return None
        
    except Exception as e:
        print(f"  Error fetching NIST data for {smiles}: {e}")
        return None


def plot_comparison(wavelengths_full: np.ndarray, predicted_full: np.ndarray,
                   wavelengths_overlap: np.ndarray, predicted_overlap: np.ndarray, 
                   experimental_overlap: np.ndarray, smiles: str, 
                   output_path: str, metrics: Dict[str, float]):
    """Plot predicted vs experimental spectra, matching the style of plot.py"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: overlaid spectra (matching plot.py style)
    # Plot full predicted spectrum (220-400nm) in light blue
    ax1.plot(wavelengths_full, predicted_full, 'b-', alpha=0.4, linewidth=2, 
             label='Predicted (full range)', zorder=1)
    
    # Plot predicted spectrum in overlap region (darker blue)
    ax1.plot(wavelengths_overlap, predicted_overlap, 'b-', linewidth=3, 
             label='Predicted (overlap)', zorder=2)
    
    # Plot experimental spectrum in overlap region
    ax1.plot(wavelengths_overlap, experimental_overlap, 'r--', linewidth=3, 
             label='NIST Experimental', zorder=3)
    
    # Style matching plot.py
    ax1.axvline(x=wavelengths_full[0], color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylim([-0.1, 1.2])
    ax1.set_xlim([210, 410])
    ax1.set_xlabel('Wavelength (nm)', fontsize=20)
    ax1.set_ylabel('Absorbance', fontsize=20)
    ax1.set_title(f'UV-Vis Spectrum Comparison\n{smiles}', fontsize=12)
    ax1.legend(loc='best', fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: difference (only in overlap region)
    diff = predicted_overlap - experimental_overlap
    ax2.plot(wavelengths_overlap, diff, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Wavelength (nm)', fontsize=20)
    ax2.set_ylabel('Difference (Pred - Exp)', fontsize=20)
    ax2.set_title('Residual Difference (Overlap Region Only)', fontsize=12)
    ax2.tick_params(labelsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([wavelengths_overlap[0] - 5, wavelengths_overlap[-1] + 5])
    
    # Add metrics text box
    metrics_text = f"R² = {metrics['r_squared']:.3f}\n"
    metrics_text += f"Pearson r = {metrics['pearson_r']:.3f}\n"
    metrics_text += f"MAE = {metrics['mae']:.3f}\n"
    metrics_text += f"RMSE = {metrics['rmse']:.3f}\n"
    metrics_text += f"Overlap: {wavelengths_overlap[0]:.0f}-{wavelengths_overlap[-1]:.0f} nm"
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted UV-Vis spectra with experimental NIST data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_predicted_vs_nist_spectra.py predicted_spectra.csv
  python compare_predicted_vs_nist_spectra.py predicted_spectra.csv -o results.csv --plots-dir plots/
  python compare_predicted_vs_nist_spectra.py predicted_spectra.csv --force-log-epsilon --no-plots

Input CSV format:
  smile,{normalized wavelength values 0-1}
  C1=CC=CC=C1,0.01,0.05,0.12,...
        """
    )
    
    parser.add_argument('input_file', 
                       help='CSV file with predicted spectra (SMILES and normalized values)')
    parser.add_argument('-o', '--output', 
                       default='nist_comparison_results.csv',
                       help='Output CSV file for comparison metrics (default: nist_comparison_results.csv)')
    parser.add_argument('-p', '--plots-dir', 
                       default='comparison_plots',
                       help='Directory for saving comparison plots (default: comparison_plots)')
    parser.add_argument('--no-plots', 
                       action='store_true',
                       help='Skip generating comparison plots')
    parser.add_argument('--force-log-epsilon', 
                       action='store_true',
                       help='Force conversion from log-epsilon scale for NIST data')
    parser.add_argument('--max-results', 
                       type=int, 
                       default=5,
                       help='Maximum number of NIST search results to check (default: 5)')
    parser.add_argument('--min-wavelength',
                       type=float,
                       default=220.0,
                       help='Minimum wavelength in nm (default: 220.0)')
    parser.add_argument('--max-wavelength',
                       type=float,
                       default=400.0,
                       help='Maximum wavelength in nm (default: 400.0)')
    
    args = parser.parse_args()
    
    # Create plots directory if needed
    if not args.no_plots:
        Path(args.plots_dir).mkdir(exist_ok=True)
    
    print(f"Reading predicted spectra from: {args.input_file}")
    print(f"Output will be written to: {args.output}")
    if not args.no_plots:
        print(f"Plots will be saved to: {args.plots_dir}/")
    print()
    
    # Read predicted spectra
    with open(args.input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        
        # Wavelengths are simply 220-400nm in order
        # Number of columns minus 1 (for SMILES column) gives us number of wavelength points
        num_wavelengths = len(header) - 1
        wavelengths = np.linspace(220, 400, num_wavelengths)
        
        print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
        print(f"Number of wavelength points: {len(wavelengths)}")
        print()
        
        # Prepare output
        results = []
        
        for row in reader:
            smiles = row[0]
            predicted_spectrum = np.array([float(x) for x in row[1:]])
            
            print(f"Processing: {smiles}")
            
            # Fetch NIST data
            nist_data = fetch_nist_spectrum(smiles, args.max_results, args.force_log_epsilon)
            
            if nist_data is None:
                print(f"  Skipping - no NIST data available\n")
                results.append({
                    'smiles': smiles,
                    'nist_available': False,
                    'r_squared': np.nan,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'spectral_angle': np.nan,
                    'overlap_coefficient': np.nan
                })
                continue
            
            nist_wavelengths, nist_absorbances = nist_data
            
            # Normalize both spectra
            predicted_norm = normalize_spectrum(wavelengths, predicted_spectrum)
            nist_norm = normalize_spectrum(nist_wavelengths, nist_absorbances)
            
            # Debug: Print wavelength ranges
            print(f"  Predicted wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm ({len(wavelengths)} points)")
            print(f"  NIST wavelength range: {nist_wavelengths.min():.1f} - {nist_wavelengths.max():.1f} nm ({len(nist_wavelengths)} points)")
            
            # Determine overlapping wavelength range
            overlap_min = max(wavelengths[0], nist_wavelengths.min())
            overlap_max = min(wavelengths[-1], nist_wavelengths.max())
            
            if overlap_min >= overlap_max:
                print(f"  ERROR: No wavelength overlap between predicted and NIST data!")
                print(f"  Skipping comparison\n")
                results.append({
                    'smiles': smiles,
                    'nist_available': False,
                    'r_squared': np.nan,
                    'mae': np.nan,
                    'rmse': np.nan,
                    'spectral_angle': np.nan,
                    'overlap_coefficient': np.nan
                })
                continue
            
            print(f"  Wavelength overlap: {overlap_min:.1f} - {overlap_max:.1f} nm")
            
            # Filter both spectra to only overlapping region
            pred_mask = (wavelengths >= overlap_min) & (wavelengths <= overlap_max)
            wavelengths_overlap = wavelengths[pred_mask]
            predicted_overlap = predicted_norm[pred_mask]
            
            print(f"  Predicted points in overlap: {np.sum(pred_mask)} / {len(wavelengths)}")
            print(f"  Unused predicted points: {len(wavelengths) - np.sum(pred_mask)}")
            
            # Interpolate NIST data to match predicted wavelength grid (only in overlap region)
            nist_interp = interpolate_spectrum(nist_wavelengths, nist_norm, wavelengths_overlap)
            
            print(f"  Predicted spectrum range (overlap): {predicted_overlap.min():.3f} - {predicted_overlap.max():.3f}")
            print(f"  NIST interpolated range (overlap): {nist_interp.min():.3f} - {nist_interp.max():.3f}")
            
            # Calculate metrics on overlapping region only
            metrics = calculate_metrics(predicted_overlap, nist_interp)
            
            print(f"  R² = {metrics['r_squared']:.3f}, Pearson r = {metrics['pearson_r']:.3f}, MAE = {metrics['mae']:.3f}, RMSE = {metrics['rmse']:.3f}\n")
            
            # Plot comparison if requested
            if not args.no_plots:
                plot_filename = f"{args.plots_dir}/{smiles.replace('/', '_').replace('\\', '_')[:50]}.png"
                # Pass both full and overlap data to plotting function
                plot_comparison(wavelengths, predicted_norm, 
                              wavelengths_overlap, predicted_overlap, nist_interp, 
                              smiles, plot_filename, metrics)
            
            # Store results
            results.append({
                'smiles': smiles,
                'nist_available': True,
                **metrics
            })
    
    # Write results to CSV
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['smiles', 'nist_available', 'r_squared', 'pearson_r', 'mae', 'rmse', 
                     'spectral_angle', 'overlap_coefficient']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    available_results = [r for r in results if r['nist_available']]
    
    if len(available_results) > 0:
        avg_r2 = np.nanmean([r['r_squared'] for r in available_results])
        avg_pearson = np.nanmean([r['pearson_r'] for r in available_results])
        avg_mae = np.nanmean([r['mae'] for r in available_results])
        avg_rmse = np.nanmean([r['rmse'] for r in available_results])
        
        print(f"Molecules with NIST data: {len(available_results)} / {len(results)}")
        print(f"Average R²: {avg_r2:.3f}")
        print(f"Average Pearson r: {avg_pearson:.3f}")
        print(f"Average MAE: {avg_mae:.3f}")
        print(f"Average RMSE: {avg_rmse:.3f}")
    else:
        print("No molecules with available NIST data")
    
    print(f"\nResults saved to: {args.output}")
    if not args.no_plots:
        print(f"Plots saved to: {args.plots_dir}/")


if __name__ == '__main__':
    main()
