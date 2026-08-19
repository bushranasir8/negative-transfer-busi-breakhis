# Negative transfer from histopathology to breast ultrasound

Aligning unlabelled breast histopathology with breast ultrasound by domain-adversarial training
does not help. It hurts, by ten accuracy points. This repository holds the code and the result
artifacts behind that claim.

> **Negative Transfer from Histopathology to Breast Ultrasound Classification:
> A Controlled Ablation of Domain-Adversarial Alignment**
> Bushra Nasir\*, Ahmad Muzaffar Khan\*, Hamid Muzaffar Khan, Kashif Zafar
> \*These authors contributed equally.

## The result

Two configurations, identical in every respect except the domain-adversarial term. Same three
backbones, same split, same schedule, same seed, same evaluation. Only the gradient reversal
layer and its unlabelled BreaKHis stream differ.

| Configuration | Adversarial term | Test accuracy | Macro-F1 |
|---|:---:|---:|---:|
| HViTE-U | no | **79.37%** | **0.7732** |
| HViTE-U + DA | yes | 69.37% | 0.5099 |
| **Effect of alignment** | | **−10.00 pts** | **−0.2633** |
| SSDAVT (different architecture, also adversarial) | yes | 58.13% | 0.3078 |
| Majority-class baseline (91 benign of 160) | — | 56.88% | 0.2417 |

The aligned model does not merely underperform. Its recall on the *normal* class falls to
0.0741, and SSDAVT's to zero: alignment pushes the classifier toward the majority class. The
per-class breakdown is in `results/`.

## What this repository contains

| Path | What it is |
|---|---|
| `cross_domain_experiments.ipynb` | The complete experimental notebook, stored with its outputs: data loading, the SSDAVT model, the HViTE-U ensemble with and without the domain-adversarial term, the SimCLR pretraining run on the pooled corpus, and a cross-domain episodic routine. Executable code and stored outputs are exactly as they were run; only comments were tidied for readability. |
| `results/cd_msvte_u_test_report.txt` | Per-class test report for the **adversarially trained** HViTE-U configuration (69.37% accuracy, macro-F1 0.5099). |
| `results/ssdavt_test_report.txt` | Per-class test report for SSDAVT (58.13% accuracy, macro-F1 0.3078). |
| `results/cd_hcml_simclr_loss.png` | SimCLR pretraining loss on the pooled BreaKHis + BUSI corpus. |
| `results/ssdavt_losses.png`, `results/ssdavt_val_acc.png` | SSDAVT training curves. |
| `requirements.txt`, `CITATION.cff`, `LICENSE` | Dependencies, citation metadata, MIT license. |

## Names in the code vs. names in the paper

The notebook uses the working names the experiments were run under; the paper uses the names
introduced in its Methods section. They refer to the same configurations:

| In the notebook | In the paper |
|---|---|
| `MSVTE-U`, BUSI-only (`train_msvte_u_bus_only`) | **HViTE-U** — the baseline arm |
| `CD-MSVTE-U`, cross-domain (`train_cd_msvte_u`) | **HViTE-U + DA** — the adversarial arm |
| `SSDAVT` (`train_ssdavt`) | SSDAVT |
| `CD-HCML`, SimCLR stage (`train_simclr_encoder_cross_domain`) | Contrastive pretraining |
| `CD-HCML`, ProtoNet stage (`train_cd_hcml_protonet`) | Not reported — see *Defects and limitations* below |

## Reproducing the ablation

Cells 2 to 17 are definitions; cells 18 to 20 are the drivers. Run every definition cell in
order, then call the routine you want:

```python
train_msvte_u_bus_only()              # baseline arm    -> HViTE-U
train_cd_msvte_u()                    # adversarial arm -> HViTE-U + DA
train_ssdavt()                        # the separate SSDAVT architecture
train_simclr_encoder_cross_domain()   # contrastive pretraining on the pooled corpus
```

The notebook as executed ran `train_cd_msvte_u()` (cell 19) and the episodic routine (cell 20);
`train_ssdavt()` in cell 18 is commented out, and the baseline arm was run separately. Each
routine writes its report into `PROJECT_ROOT/results`.

What makes the comparison controlled: `build_msvte_models` and `build_cd_msvte_models` return
the same three backbones — `vit_small_patch16_224.augreg_in21k`,
`vit_base_patch16_224.augreg_in21k` and `deit_small_patch16_224` — with the same dropout, and
both arms are evaluated through the same `cd_msvte_mc_ensemble_predict` with 10 Monte Carlo
dropout passes. The adversarial arm differs only by `DomainAdaptiveViTClassifier`, which adds
the gradient reversal layer and the domain head, and by the unlabelled BreaKHis stream that
feeds it.

Settings are all in the `CFG` class in cell 2: seed 42, 224×224 inputs, 30 epochs for both
HViTE-U arms, 50 for SSDAVT, 20 for SimCLR.

## Data

Neither dataset is redistributed here. Both are public:

- **BUSI** — Breast Ultrasound Images dataset (Al-Dhabyani et al., 2020), 3 classes: benign,
  malignant, normal. The 160-image test split holds 91 benign, 42 malignant, 27 normal.
- **BreaKHis** — Breast Cancer Histopathological Database (Spanhol et al., 2016). In the
  controlled ablation, in SSDAVT and in the contrastive pretraining stage it is used as an
  **unlabelled** auxiliary domain and its class labels are discarded on load. The episodic
  routine described below is the sole exception, and its output is not reported.

Download both, then set the paths at the top of the notebook's configuration cell:

```python
DATA_ROOT     = Path("<your data directory>")
BREAKHIS_ROOT = DATA_ROOT / "breakhis" / "BreaKHis_v1" / "histology_slides" / "breast"
BUSI_ROOT     = DATA_ROOT / "busi" / "Dataset_BUSI_with_GT" / "Original"
PROJECT_ROOT  = Path("<where checkpoints and results should be written>")
```

Those are the only paths that need changing. The stored outputs still print the original
machine's absolute paths under `D:\My Thesis\...`; they are left as they were written rather
than edited after the fact.

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

Exact versions, the GPU model and wall-clock times were not recorded at run time, so nothing is
pinned or claimed here. `timm` supplies the ViT-S/16, ViT-B/16, DeiT-S/16 and DINO ViT-S/16
backbones.

## What is not in this release

- **The baseline classification report.** The headline baseline figures — HViTE-U without the
  adversarial term, 79.37% accuracy and 0.7732 macro-F1 — are reported from the original
  experimental record. That report file was not archived separately. The routine that produces
  it, `train_msvte_u_bus_only`, is present and will regenerate it.
- **Trained checkpoints.** The `models/` directory was not retained.
- **The episodic implementation behind the paper's Section 5.7** (48.11% cross-modality
  meta-test, ~86.6% source-domain meta-training, 80.17% ± 6.71 within-domain 3-way 5-shot).
  Those figures come from an earlier implementation that is not part of this release. The
  episodic routine that *is* in the notebook is a later and different one, and it is defective;
  see below. The paper states this.

## Defects and limitations

Everything here is also disclosed in the paper. It is repeated so that nobody reading the code
has to discover it for themselves.

**The cross-domain episodic routine is defective and its output is not used.**
`train_cd_hcml_protonet` builds Prototypical Network support sets from BreaKHis and query sets
from BUSI. Two defects:

1. It selects BreaKHis support classes with the filter `int(y) in (0, 1)`. Because the loader
   registers benign subtypes first, indices 0 and 1 are *adenosis* and *fibroadenoma* — both
   **benign**. The prototype intended to represent the malignant class is therefore built from
   benign histopathology images.
2. The prototypical head is trained with cross-entropy against **BUSI query labels**, so the
   configuration is supervised in the target domain. It is not the label-free cross-modality
   transfer it appears to be, and its accuracy is not comparable to a transfer result.

The result file this routine wrote was removed from the repository rather than shipped with a
number that means nothing. The routine itself is left in place because the notebook is released
as executed.

**A wrong heading in a result file.** `cd_msvte_u_test_report.txt` prints its uncertainty
statistics under the heading `Uncertainty (entropy) stats`. That heading is wrong. The quantity
computed and written is the **predictive variance** of the MC-dropout probability estimates
(`probs_mc.var(dim=0).mean(dim=1)`), which is what the paper defines and reports.

**A name bound twice.** Cell 11 binds `train_msvte_u_bus_only` and `build_msvte_models` a second
time each. The later definitions are the ones that take effect and they are the complete ones;
the earlier `train_msvte_u_bus_only` is a three-line stub calling a routine that does not exist.
It is dead code, left in place because the notebook is released as executed.

**Limitations of the study itself.**

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

## How to cite

Cite the paper. Machine-readable metadata for this repository is in `CITATION.cff`, which GitHub
renders through the **Cite this repository** button in the sidebar.

## License

MIT, see `LICENSE`. The datasets carry their own licenses and are not covered by it.
