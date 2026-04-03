# GC-MS Deconvolution

A failed experiment at GC-MS deconvolution from scratch — extracting overlapping molecules from gas chromatography mass spectrometry data. Works great on synthetic data, not so much on real data.

The idea: use [MCR-ALS](https://doi.org/10.1039/c4ay00571f) (Multivariate Curve Resolution — Alternating Least Squares) to separate overlapping peaks, with a component count estimator trained on synthetic data. The MCR-ALS implementation follows the textbook approach described in de Juan, Jaumot & Tauler (2014), *"Multivariate Curve Resolution (MCR). Solving the mixture analysis problem"*, Anal. Methods, 6, 4964–4976.

**Blog:** [gcms.jonasberdoz.dev](https://gcms.jonasberdoz.dev)

## What's in here

- **Part 0:** Why deconvolution matters — overlapping molecules contaminate spectra and break identification
- **Part 1:** Building a synthetic data generator from real elution profiles and MassBank spectra
- **Part 2:** Counting components with SVD + RandomForest (98.5% accuracy on synthetic data)
- **Part 3:** MCR-ALS for recovering elution profiles and spectra
- **Part 4:** Applying the pipeline to real data (it doesn't work well — yet)

## Project structure

```
gcms/                     # Core package
  preprocessing.py        # Gaussian denoising + AsLS baseline removal
  peak_picking.py         # TIC peak detection with edge refinement
  estimator.py            # Component count estimation (SVD + RandomForest)
  mcr.py                  # MCR-ALS (Multivariate Curve Resolution)
  identification.py       # Cosine similarity search against spectra library
  pipeline.py             # Full pipeline tying everything together

tools/                    # CLI tools and scripts
  run_pipeline.py         # Run the full pipeline on an analysis
  viewer.py               # Interactive Panel viewer for results
  benchmark_mcr.py        # MCR-ALS benchmark on synthetic data
  generate_dataset.py     # Generate synthetic GC-MS datasets
  train_model.py          # Train component count estimator

data/                     # Not in repo — see below
  spectra.json            # 9,971 EI mass spectra from MassBank
  A0/, B0/, ...           # Real GC-MS analyses (Copenhagen Soft Camel Cheese)
  synthetic_peaks/        # 1,000 synthetic samples for training
  synthetic_peaks_mcr/    # 1,000 synthetic samples for MCR-ALS benchmark

models/
  component_counter/      # Trained RandomForest model

posts/                    # Blog HTML (deployed to Vercel)
build_blog.py             # Blog generator with embedded Bokeh plots
```

## Usage

### Run the pipeline on an analysis

```bash
uv run python tools/run_pipeline.py data/A0
```

Outputs `ms_clean.npy` + `peaks.json` to `data/A0/results/<timestamp>/`.

### View results interactively

```bash
uv run panel serve tools/viewer.py --show --args data/A0/results/<timestamp>
```

### Build the blog

```bash
uv run python build_blog.py
```

## Data download

[Download data.zip from Google Drive](https://drive.google.com/file/d/1m30gxfcLzprSg9v74dPD0UIZaQjrAu0X/view?usp=sharing) (1.6 GB), then:

```bash
unzip data.zip
```

## Data sources

- **GC-MS data:** [Copenhagen Soft Camel Cheese dataset](https://ucphchemometrics.com/)
- **Mass spectra:** [MassBank](https://github.com/MassBank/MassBank-data) (9,971 EI spectra)

## Built with

Python, NumPy, SciPy, scikit-learn, Bokeh, Panel, Pydantic. Almost entirely built with [Claude Code](https://claude.ai/code).
