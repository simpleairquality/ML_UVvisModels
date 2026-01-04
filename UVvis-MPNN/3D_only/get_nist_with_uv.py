#!/usr/bin/env python3
"""
Extract SMILES for all compounds in NIST WebBook that have UV-Vis spectra.

Usage:
    python extract_nist_smiles_with_uv.py -o nist_compounds_with_uv.txt

This script will:
1. Get all compounds from NIST using nistchempy.get_all_data()
2. For each compound ID, check if it has UV-Vis spectra
3. Extract SMILES representation
4. Save to output file (one SMILES per line, no header)
"""

import argparse
import sys
import time
import pandas 
import nistchempy as nist
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

def get_smiles_from_compound(inchi):
    """
    Extract SMILES from a NIST compound object.
    
    Args:
        compound: nistchempy compound object
        
    Returns:
        SMILES string or None
    """
    mol = Chem.inchi.MolFromInchi(inchi)
    return Chem.MolToSmiles(mol)


def has_conformer(smile):
    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    if AllChem.EmbedMolecule(mol) != 0:
        return False

    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for s in Chem.FindPotentialStereo(mol):
        if s.type == rdchem.StereoType.Atom_Tetrahedral and \
           s.specified == rdchem.StereoSpecified.Unspecified:
            return False

    mol.GetConformer()
    return True


def extract_compounds_with_uv_spectra(output_file: str, delay: float = 0.5):
    """
    Extract SMILES for compounds with UV-Vis spectra from NIST.
    
    Args:
        output_file: Output text file for SMILES (one per line, no header)
        delay: Delay between requests in seconds
    """
    
    print("NIST WebBook SMILES Extractor for Compounds with UV-Vis Spectra")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Delay between requests: {delay}s")
    print()
    
    print("Fetching all compounds from NIST WebBook...")
    
    try:
        # Get all compounds from NIST - returns a pandas DataFrame
        df = nist.get_all_data()
        
        print(f"Retrieved {len(df)} compounds from NIST")
        print("-" * 70)
        
    except Exception as e:
        print(f"Error fetching compounds from NIST: {e}")
        import traceback
        traceback.print_exc()
        return
    
    compounds_with_uv = 0
    compounds_processed = 0
    
    # Open output file (no header, just SMILES)
    with open(output_file, 'w') as f:
        
        for idx, row in df.iterrows():
            compounds_processed += 1
            
            try:
                # Get the compound ID from the dataframe
                compound_id = row['ID']
                if pandas.isna(row['UV/Visible spectrum']) or pandas.isna(row['inchi']):
                    continue

                # Compound has UV spectra - extract SMILES
                smiles = get_smiles_from_compound(row['inchi'])
              
                if not has_conformer(smiles):
                    continue

                if smiles:
                    compounds_with_uv += 1
                  
                    # Write to file (just SMILES, no header)
                    f.write(f"{smiles}\n")
                    f.flush()
                
            except Exception as e:
                # Error processing this compound, skip it
                print(f"  Error processing compound {row['ID']}: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total compounds processed: {compounds_processed}")
    print(f"Compounds with UV spectra: {compounds_with_uv}")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMILES for NIST compounds with UV-Vis spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_nist_smiles_with_uv.py -o nist_uv_compounds.txt
  python extract_nist_smiles_with_uv.py -o nist_uv_compounds.txt --delay 1.0

Output format: One SMILES per line, no header.
Use Ctrl+C to stop early and save progress.
        """
    )
    
    parser.add_argument('-o', '--output',
                       required=True,
                       help='Output text file for SMILES (one per line, no header)')
    parser.add_argument('--delay',
                       type=float,
                       default=0.5,
                       help='Delay between NIST requests in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    extract_compounds_with_uv_spectra(args.output, args.delay)


if __name__ == '__main__':
    main()
