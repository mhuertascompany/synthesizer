# AstroPT DESI representation workflow

The supplied checkpoint was trained with a `DESISpectrum` modality containing
7781 flux samples, grouped into tokens of width 10. Its metadata specifies
`asinh` normalization, a 768-dimensional backbone, and a final CLS token.

The original DESI FITS files are never modified. Preparation writes derived
NPZ files containing the common wavelength grid, coadded flux and inverse
variance, normalized tokens, padding mask, and provenance.

## 1. Prepare mock spectra

```bash
python prepare_astropt_desi.py \
  --input-root /u/mhuertas/data/euclid/tngmatched_euclid_desi \
  --output-dir ./astropt/desi_inputs \
  --limit 10 \
  --overwrite
```

This combines the B/R/Z cameras by inverse variance on the standard
3600--9824 Angstrom grid with 0.8 Angstrom spacing. The result has 779 tokens;
the final token contains the last sample followed by nine padding values.

## 2. Extract DESI-only embeddings

Clone the checkpoint's source branch separately, then run:

```bash
python extract_astropt_desi_embeddings.py \
  --checkpoint ./astropt/ckpt_best.pt \
  --astropt-source /u/mhuertas/python/astroPT \
  --manifest ./astropt/desi_inputs/manifest.csv \
  --output ./astropt/desi_embeddings_mocks.npz \
  --device cuda \
  --batch-size 32
```

For a CPU compatibility test, use `--device cpu --batch-size 1` and prepare a
single spectrum. The checkpoint is trusted local input and is loaded with
PyTorch pickle support because it embeds its `ModalityRegistry` object.

The public `victor_branch` predates the checkpoint's explicit CLS and modality
embedding layer. The extractor reconstructs that DESI-only inference path from
the checkpoint tensors, while requiring an exact match for every public base
layer. It aborts on any architectural mismatch instead of silently loading a
partial model.

## 3. Compare against observed DESI

Run the same preparation and extraction commands on a held-out observed DESI
sample, then compare both embedding files:

```bash
python compare_astropt_desi_embeddings.py \
  --mocks ./astropt/desi_embeddings_mocks.npz \
  --observed ./astropt/desi_embeddings_observed.npz \
  --output-prefix ./astropt/desi_mock_vs_observed
```

The comparison reports a cross-validated mock/observed classifier AUC, nearest
cosine similarities, RBF MMD, and a joint PCA diagnostic. AUC near 0.5 means
the two domains are difficult to distinguish; a high AUC indicates a strong
simulation-to-observation representation gap.
