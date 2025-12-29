# UVvis-MPNN Scripts

This directory contains the files necessary to run the 4 variations of the UVvis-MPNN model. They are broken into the following:

1. `3D_distance_plus_spectra` :: Model that was trained on 3D distances between atoms and DFT UV spectra as features
2. `3D_only` :: Model that was trained on 3D distances between atoms as features
3. `original` :: Model that was trained only on the 2D SMILES strings as features
4. `original_plus_spectra` :: Model that was trained on 2D SMILES and DFT UV spectra as features

The pre-trained weights for these models are located in their respective directories. Based on the architecture of `Chemprop`, 10 model weights are ensembled and used for prediction.

The method of prediction is straight forward, once in the respective models main folder:

```bash
python predict.py --test_path smiles.csv --features_path features.csv --checkpoint_dir model_checkpoints --preds_path uv_preds.csv
```

**Note**: This `--features_path` is for the models that incorporate DFT spectra as a training feature (`_plus_spectra` models). It is not explicity needed, but may improve prediction results.

The SMILES input, `smiles.csv` should be a csv list of Strings with the header: `smiles`. The Features input `features.csv` should be csv list of absorption values corresponding to the provided SMILES with the header being some index corresponding to the wavelength (i.e, 220->400). 


This input organization follows the training process, which takes a csv of smiles and absorption values and then splits them into 2 separate files.

```
smiles, 1, 2, 3, ..., 181
CCC, 0.5, 0.3, 0.2, ..., 1
```

and after splitting out the smiles column:

```console
total_data.csv
├── smiles.csv
└── features.csv
```

## Scaling

After running the inference method of the UVvis-MPNN, you will probably need one last scaling of the values between 0-1. There is a provided script `spectra_scaling.py` to run a min-max scaling over the `uv_preds.csv` file. This will ensure smooth plotting and interpretable results.

## Using Pre-computed Structures

The `3D_only` and `3D_distance_plus_spectra` methods will automically compute the 3D conformer of the provided SMILES string using `RDKit`. However, if you desire to use your own DFT-optimized structures or alternative 3D conformers, you can utilize the `--structures_path` which is a string pointing to a folder where you store a series of `.xyz` files for conversion to `RDKit` mol objects.

This snippet from the codebase illustrates the internal logic:
```python
if args.structures_path is True:
            # Convert smiles to molecule
            reader = csv.DictReader(open(args.structures_path+'/smiles_to_xyz.csv'))
            xyz_map = next(reader) 
            xyz_file = xyz_map.get(smiles)
            xyz_path = args.structures_path+'/{0}'.format(xyz_file)
        
            with open(xyz_path,'r') as f:
                lines = f.read()

            mol = Chem.MolFromPDBBlock(xyz_to_pdb_block(lines), removeHs=False)
```

This means that to use your own structures you need 2 types of files:

1. An [XYZ coordinates file](https://en.wikipedia.org/wiki/XYZ_file_format) (`*.xyz`) for each molecule in your inference set
2. A mapping file named `smiles_to_xyz.csv` which maps the SMILES string of your compound to it's corresponding XYZ coordinate file

For example, if I had a SMILES string for Glucose: OCC1OC(O)C(C(C1O)O)O and it's 3D coordinates were stored in a file named `GLUCOSE.xyz`, I would need my mapping file to show:

```
OCC1OC(O)C(C(C1O)O)O, GLUCOSE.xyz
```

Then when the code generates a `MolGraph` for this compound, it will be able to locate the correct optimized structure. You only need a single mapping file containing all SMILES/XYZ pairs, but one XYZ file for each molecule.

You would then run the following to generate predictions on your generated compounds:

```bash
python predict.py --test_path smiles.csv --features_path features.csv --structures_path /path/to/structures --checkpoint_dir model_checkpoints --preds_path uv_preds.csv
```

Remember that this is only if you have pre-computed structures that are ideally DFT-optimized. In most cases, an `RDKit` 3D Conformer is good enough to get a reliable prediction.

# Notes on Running/Reproducing

We are working with the `3D_distance_plus_spectra` version to start.
To begin, there is no requirements.txt file, so we have no idea what package versions this was run or trained with.
So, we create our own.
To get started,

```bash
cd 3D_distance_plus_spectra
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next we need a `smiles_to_xyz.csv` file as well as the corresponding data files.
[In their code](https://github.com/simpleairquality/ML_UVvisModels/blob/main/UVvis-MPNN/3D_distance_plus_spectra/chemprop/features/featurization.py#L228), they have hard-coded the paths to these files to something unique to the person who wrote the code.
We have modified this to default everything into the `./smiles_to_xyz/` directory, with the CSV file defaulting to `./smiles_to_xyz/smiles_to_xyz.csv`.

It looks like these may be generated with an external tool, [smi2xyz](https://github.com/hoelzerC/smi2xyz/).
To generate this, separately clone into that repo and set it up using their instructions.

Note that `smi2xyz` is not a packge of any kind, so we merely coped the source into the `3D_distance_plus_spectra` directory.

Then we made our own tool, which you can use like this:
```
cat ../../Data/UV_w_SMILES.csv | python smiles_to_xyz.py ./smiles_to_xyz/
```

Then we can run:
```bash
venv/bin/python3 predict.py --test_path ../../Data/UV_w_SMILES.csv --checkpoint_dir models_3D_distance_plus_spectra --preds_path uv_preds.csv
```

Note that it would be nice to eventually be able to include the `--features-path` argument to improve the models, which is described further in this document.

After some additional work, we have found that the pretrained models for `3D_distance_plus_spectra` are missing multiple critical data files, `/UV_chemprop_Data_both_DFT_expt.csvv` (sic) and `DFT_spectra_feature_chemprop.csv`.
Lacking these files, this path is dead in the water, but we can fall back to the `3D_only`, and apply many of these lessons.

Let's proceed like that.

```bash
cd 3D_only
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Follow the instructions as before to create the `smiles_to_xyz` data, and put it in the `3D_only` directory.

Now:

```bash
venv/bin/python3 predict.py --test_path ../../Data/UV_w_SMILES.csv --checkpoint_dir models_3D_only --preds_path uv_preds.csv
```

And we get results in `uv_preds.csv` of the same directory!
Now, as they say, let's try `spectra_scaling`.

```bash
venv/bin/python ../spectra_scaling.py uv_preds.csv
```

outputs `scaled_spectra.csv`

Two things stand out.
1)  These normalized spectra can't be simple absorbances.  These must be relative to the other computed absorbances, so they will have to be calibrated to real, known spectra.
2)  What units are the header in?  Microns I guess?  But, it's odd that you would choose microns as a feature for something that's usually in nanometers.

Running a slightly modified `plot.py`, 

```bash
venv/bin/python plot.py
```

I think reveals what was intended by the header, which is deeply confusing but at least clarified here.
0 is like 210nm, and 1 is 400nm.  

I wonder how well this agrees with a real absorbance spectrum.  
Let's try our best buddy forever, Styrene.


