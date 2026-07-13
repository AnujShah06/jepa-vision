# Terminal Benchmark — Val Split (Dry Run)

split=val  dry_run=True  K=8  n_boot=2000
corruptions: ['gaussian_noise', 'defocus_blur', 'jpeg_compression']
severities:  [1, 3, 5]

**MAE trained: INCLUDED**
**hardmask_s0\*: single seed, REJECTED lever (R1)**

---

## Stage 2 — Corruption AUROC (point, mean over severities)

Model               gaussian_nois defocus_blur  jpeg_compress 
--------------------------------------------------------------
ref_s0              0.783         0.234         0.604         
ref_s1              0.676         0.331         0.624         
ref_s2              0.659         0.276         0.584         
hardmask_s0*        0.878         0.256         0.602         
pixel_std           0.742         0.298         0.484         
random_init         0.478         0.519         0.501         
mahal_tgt           0.739         0.933         0.569         
mahal_ctx           0.856         0.974         0.664         
mae_untrained       0.448         0.518         0.512         
mae_trained         0.959         0.054         0.539         

## Stage 2 detail — ref_s0 per-severity

Corruption            sev1   sev3   sev5     mean
---------------------------------------------------
gaussian_noise        0.635  0.767  0.948    0.783
defocus_blur          0.341  0.228  0.133    0.234
jpeg_compression      0.545  0.594  0.671    0.604

## Stage 2 detail — ref_s0 bootstrap CIs

Corruption              Sev   Point  95% CI
----------------------------------------------------
gaussian_noise            1   0.635  [0.610, 0.660]
gaussian_noise            3   0.767  [0.744, 0.790]
gaussian_noise            5   0.948  [0.935, 0.960]
defocus_blur              1   0.341  [0.317, 0.366]
defocus_blur              3   0.228  [0.208, 0.249]
defocus_blur              5   0.133  [0.118, 0.149]
jpeg_compression          1   0.545  [0.519, 0.571]
jpeg_compression          3   0.594  [0.569, 0.620]
jpeg_compression          5   0.671  [0.648, 0.695]

## Stage 3 — OOD AUROC

Model                     SVHN    CIFAR-10
--------------------------------------------
ref_s0                   0.098       0.411
ref_s1                   0.133       0.518
ref_s2                   0.088       0.438
hardmask_s0*             0.165       0.401
pixel_std                0.102       0.405
random_init              0.572       0.515
mahal_tgt                0.985       0.855
mahal_ctx                0.998       0.949
mae_untrained            0.716       0.519
mae_trained              0.013       0.147

## Stage 4 — Probe Grid (locked: target mean+zscore, lr-sweep, 200ep, 3 probe seeds)

Model                         n=40         n=200         n=400         n=4000
---------------------------------------------------------------------------
ref_s0                0.2937±0.0034  0.4147±0.0246  0.4550±0.0073  0.6030±0.0008
ref_s1                0.2690±0.0043  0.3653±0.0116  0.4170±0.0118  0.5643±0.0017
ref_s2                0.2683±0.0184  0.3897±0.0160  0.4357±0.0109  0.5803±0.0005
hardmask_s0*          0.2950±0.0022  0.4180±0.0142  0.4813±0.0146  0.5897±0.0009

JEPA ref mean                0.277         0.390         0.436         0.583
Scratch A3 mean              0.240         0.353         0.389         0.579
Gap (JEPA-ref−A3)         +0.0373*      +0.0366*      +0.0466*      +0.0039*

* A3 recipe underfit: batch=min(256,n) vs 1.5d batch=128; no augmentation vs RandomResizedCrop+HFlip. Gap shown for reference; binding gap requires recipe-fixed rerun. Gate 1B(iii) post-hoc on val; test set never reopens.

### Stage 4 detail — per probe seed

Model/seed                    n=40   n=200   n=400   n=4000
------------------------------------------------------------
ref_s0 s=0                  0.2970  0.4350  0.4480  0.6030
ref_s0 s=1                  0.2890  0.3800  0.4650  0.6020
ref_s0 s=2                  0.2950  0.4290  0.4520  0.6040
ref_s1 s=0                  0.2730  0.3720  0.4010  0.5660
ref_s1 s=1                  0.2630  0.3490  0.4210  0.5620
ref_s1 s=2                  0.2710  0.3750  0.4290  0.5650
ref_s2 s=0                  0.2900  0.3920  0.4210  0.5800
ref_s2 s=1                  0.2450  0.3690  0.4470  0.5810
ref_s2 s=2                  0.2700  0.4080  0.4390  0.5800
hardmask_s0* s=0            0.2980  0.4370  0.4660  0.5890
hardmask_s0* s=1            0.2930  0.4030  0.4770  0.5910
hardmask_s0* s=2            0.2940  0.4140  0.5010  0.5890

---

## Wall-clock summary

Stage 2 (corruption grid):    157s
Stage 3 (OOD):                604s
Stage 4 (probe grid):         191s
Total:                        952s (0.26h)