# ResNet vs Plain on CIFAR-10 — 수시평가 발표용 실험

Fork of [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
(BSD-2-Clause, © 2018 Yerlan Idelbayev). The model structure is unchanged; see the commit history for every change.

## What we changed
| # | Change | File |
|---|---|---|
| 1 | `residual` flag: `False` = plain net (same layers & params, `out += self.shortcut(x)` skipped) | `resnet.py` |
| 2 | LR schedule per **iteration**, as in the paper: 0.1, ÷10 at 50 % / 75 %, stop at 64k iters (orig.: `MultiStepLR([100,150])` per epoch, hard-coded) | `trainer.py` |
| 3 | Weight-gradient L2 norm of every conv layer (stem + each block conv1/conv2), epoch mean + value at init | `trainer.py` |
| 4 | `log.csv` / `grads.csv` per run, full checkpoint every epoch, auto-resume (orig. `--resume` was broken) | `trainer.py` |
| + | fixed seed, single GPU (no DataParallel), optional `--amp`, ResNet-110/1202 warm-up actually ends | `trainer.py` |
| + | pretrained-weight check, plots & results table, Colab notebook | `eval_pretrained.py`, `analyze.py`, `colab_run.ipynb` |

Unchanged from akamaster: data augmentation (pad 4 + random crop, h-flip), normalization, SGD (momentum 0.9, wd 1e-4), batch 128, model code.

## Experiment
- Models: plain-20, plain-56, ResNet-20, ResNet-56, seed 0.
- 64,000 iterations × batch 128 = 163.7 epochs on 50k images (paper gives iterations, not epochs).
- Trained on all 50k training images; test error measured on the 10k test set every epoch.
  The paper chose its schedule on a 45k/5k split; akamaster does not use a validation split either.
- We report the **final-epoch** test error (main) and the best test error (only to compare with akamaster's README, which reports best).

## Run
Open `colab_run.ipynb` in Colab (GPU), set `REPO`, run cells top to bottom. Results go to Google Drive
`MyDrive/resnet_cifar/{runs,figures}`. After a disconnect, re-run cells 1–3 and 5: training resumes.

Local: `python trainer.py -a resnet56 --plain --save-dir runs` then `python analyze.py --runs runs --out figures`.

## References
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016, arXiv:1512.03385 (Sec 4.2, Fig. 6, Table 6)
- K. He et al., *Identity Mappings in Deep Residual Networks*, ECCV 2016, arXiv:1603.05027 (Eq. 5)
