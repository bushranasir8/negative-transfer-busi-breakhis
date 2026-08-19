# Negative transfer from histopathology to breast ultrasound

Code and result artifacts for the paper:

> **Negative Transfer from Histopathology to Breast Ultrasound Classification:
> A Controlled Ablation of Domain-Adversarial Alignment**
> Bushra Nasir\*, Ahmad Muzaffar Khan\*, Hamid Muzaffar Khan, Kashif Zafar
> \*These authors contributed equally.

## What this repository contains

| Path | What it is |
|---|---|
| `cross_domain_experiments.ipynb` | The complete experimental notebook, stored with its outputs: data loading, the SSDAVT model, the HViTE-U ensemble with and without the domain-adversarial term, the SimCLR pretraining run on the pooled corpus, and a cross-domain episodic routine. Executable code and stored outputs are exactly as they were run; only comments were tidied for readability. |
| `results/cd_msvte_u_test_report.txt` | Per-class test report for the **adversarially trained** HViTE-U configuration (69.37% accuracy, macro-F1 0.5099). |
| `results/ssdavt_test_report.txt` | Per-class test report for SSDAVT (58.13% accuracy, macro-F1 0.3078). |
| `results/cd_hcml_simclr_loss.png` | SimCLR pretraining loss on the pooled BreaKHis + BUSI corpus. |
| `results/ssdavt_losses.png`, `results/ssdavt_val_acc.png` | SSDAVT training curves. |

## Names in the code vs. names in the paper

The notebook predates the paper's terminology. The mapping is:

| In the notebook | In the paper |
|---|---|
| `MSVTE-U`, BUSI-only (`train_msvte_u_bus_only`) | **HViTE-U** -- the baseline arm |
| `CD-MSVTE-U`, cross-domain (`train_single_cd_vit`) | **HViTE-U + DA** -- the adversarial arm |
| `SSDAVT` (`train_ssdavt`) | SSDAVT |
| `CD-HCML`, SimCLR stage (`train_simclr_encoder_cross_domain`) | Contrastive pretraining |
| `CD-HCML`, ProtoNet stage (`train_cd_hcml_protonet`) | Not reported -- see below |

The stored outputs print the original machine's absolute paths under `D:\My Thesis\...`. They
are left as they were written; the paths to change are listed under **Data** below.

The controlled ablation that carries the paper's conclusion — HViTE-U with and without the
adversarial term — is implemented in the notebook cells defining `SingleViTClassifier` /
`train_single_vit` (baseline arm) and `DomainAdaptiveViTClassifier` / `train_single_cd_vit`
(adversarial arm), with `cd_msvte_mc_ensemble_predict` used for evaluation in both.

## An honest note about the baseline

The headline baseline number in the paper — HViTE-U **without** the adversarial term, 79.37%
accuracy and 0.7732 macro-F1 — is reported from the original experimental record. **Its
classification report file was not archived separately and is therefore not in this
repository.** The training and evaluation routine that produces it is present in the notebook
(`train_msvte_u_bus_only`) and will regenerate it; note that the notebook as executed invokes
the adversarial and episodic routines, not this one.

Trained checkpoints are not included; the `models/` directory was not retained.

One further label discrepancy, stated in the paper and repeated here so nobody is misled:
`cd_msvte_u_test_report.txt` prints its uncertainty statistics under the heading
`Uncertainty (entropy) stats`. That heading is wrong. The quantity computed and written is the
**predictive variance** of the MC-dropout probability estimates
(`probs_mc.var(dim=0).mean(dim=1)`), which is what the paper defines and reports.

## A defect in the episodic routine, and what it means

The notebook contains a cross-domain episodic routine, `train_cd_hcml_protonet`, which builds
Prototypical Network support sets from BreaKHis and query sets from BUSI. **Its output is not
reported in the paper, and it should not be used.** Two defects:

1. It selects BreaKHis support classes with the filter `int(y) in (0, 1)`. Because the loader
   registers benign subtypes first, indices 0 and 1 are *adenosis* and *fibroadenoma* — both
   **benign**. The prototype intended to represent the malignant class is therefore built from
   benign histopathology images.
2. The prototypical head is trained with cross-entropy against **BUSI query labels**, so the
   configuration is supervised in the target domain. It is not the label-free cross-modality
   transfer it appears to be, and its accuracy is not comparable to a transfer result.

The result file this routine wrote has been removed from the repository rather than shipped
with a number that means nothing. The routine itself is left in place because this notebook is
released as executed.

The episodic results that *are* reported in the paper (Section 5.7: 48.11% cross-modality
meta-test, ~86.6% source-domain meta-training, 80.17% ± 6.71 within-domain 3-way 5-shot) come
from an earlier implementation which is **not** part of this release. The paper states this.

## Data

Neither dataset is redistributed here. Both are public:

- **BUSI** — Breast Ultrasound Images dataset (Al-Dhabyani et al., 2020), 3 classes:
  benign, malignant, normal.
- **BreaKHis** — Breast Cancer Histopathological Database (Spanhol et al., 2016). In the
  controlled ablation, in SSDAVT and in the contrastive pretraining stage it is used as an
  **unlabelled** auxiliary domain and its class labels are discarded on load. The episodic
  routine described above is the sole exception, and its output is not reported.

Download both, then set the two paths at the top of the notebook's configuration cell:

```python
DATA_ROOT     = Path("<your data directory>")
BREAKHIS_ROOT = DATA_ROOT / "breakhis" / "BreaKHis_v1" / "histology_slides" / "breast"
BUSI_ROOT     = DATA_ROOT / "busi" / "Dataset_BUSI_with_GT" / "Original"
PROJECT_ROOT  = Path("<where checkpoints and results should be written>")
```

Those are the only paths that need changing.

## Environment

Run on a single CUDA GPU under an Anaconda environment. Requirements:

```
torch
torchvision
timm
numpy
scikit-learn
matplotlib
pillow
tqdm
```

Exact versions were not recorded at run time, so none are pinned here. `timm` supplies the
ViT-S/16, ViT-B/16, DeiT-S/16 and DINO ViT-S/16 backbones.

## Known limitations, all disclosed in the paper

1. Results come from a **single seed and a single train/validation/test split**. No confidence
   intervals accompany the main ablation.
2. The weighted sampler is inverted — it oversamples the majority class and works against the
   weighted loss. It is identical in both arms, so it bounds absolute performance without
   confounding the ablation.
3. Gradient-norm clipping is applied in the adversarial arm and not in the baseline. Clipping is
   a stabiliser, so if it has any effect it favours the arm that performed worse, which makes the
   comparison conservative with respect to the paper's conclusion.
4. SimCLR pretraining is transductive: test images are present in the unlabelled pool. This does
   not touch the main ablation.
5. SSDAVT validation used the training augmentation pipeline. This affects checkpoint selection
   only; test evaluation is clean.
6. The episodic routine in the notebook is defective as described above and is not reported.

## License

MIT, see `LICENSE`. The datasets carry their own licenses and are not covered by it.
