import csv
import os
import argparse
import sys

# Try to import from the installed package or a local folder named 'smi2xyz'
try:
    from smi2xyz import Converter
except ImportError:
    print("Error: smi2xyz not found.")
    print("To install: pip install git+https://github.com/hoelzerC/smi2xyz.git")
    print("Or: Copy the 'smi2xyz' folder from GitHub into this directory.")
    sys.exit(1)

def process_smiles(input_handle, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    mapping_csv_path = os.path.join(output_dir, "smiles_to_xyz.csv")
    mapping_rows = []
    reader = csv.reader(input_handle)
    
    print(f"Reading from {input_handle.name}...")

    for index, row in enumerate(reader, start=1):
        if not row or not row[0].strip():
            continue
        
        smiles_string = row[0].strip()
        xyz_filename = f"{index}.xyz"
        xyz_full_path = os.path.join(output_dir, xyz_filename)

        try:
            # The library returns the XYZ data as a string
            xyz_data = Converter.smiles_to_xyz(smiles_string)
            
            # Write the string to the file
            with open(xyz_full_path, 'w', encoding='utf-8') as f:
                f.write("{}\n".format(len(xyz_data)))
                f.write(smiles_string + "\n")
                for r in range(len(xyz_data)):
                    f.write("{} {} {} {}\n".format(int(xyz_data[r][0]), xyz_data[r][1], xyz_data[r][2], xyz_data[r][3]))
            
            mapping_rows.append([smiles_string, xyz_filename])
            
            if index % 10 == 0:
                print(f"Processed {index} structures...")

        except Exception as e:
            print(f"Row {index}: Failed to convert {smiles_string[:15]}... | Error: {e}")

    # Write the mapping file
    with open(mapping_csv_path, mode='w', newline='', encoding='utf-8') as map_f:
        writer = csv.writer(map_f)
        writer.writerows(mapping_rows)
    
    print(f"\nDone! Saved {len(mapping_rows)} XYZ files to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert SMILES to XYZ.")
    parser.add_argument("output_dir", help="Output directory for results.")
    parser.add_argument("input_file", nargs="?", type=argparse.FileType('r'), 
                        default=sys.stdin, help="Input CSV (defaults to stdin).")
    
    args = parser.parse_args()
    process_smiles(args.input_file, args.output_dir)
