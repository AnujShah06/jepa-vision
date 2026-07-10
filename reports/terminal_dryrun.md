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
ref_s0              0.785         0.234         0.604         
ref_s1              0.677         0.331         0.624         
ref_s2              0.660         0.276         0.584         
hardmask_s0*        0.877         0.256         0.602         
pixel_std           0.742         0.298         0.484         
random_init         0.762         0.878         0.901         
mahalanobis         0.857         0.974         0.664         
mae_untrained       0.580         0.458         0.482         
mae_trained         0.959         0.054         0.539         

## Stage 2 detail — ref_s0 per-severity

Corruption            sev1   sev3   sev5     mean
---------------------------------------------------
gaussian_noise        0.639  0.768  0.948    0.785
defocus_blur          0.341  0.228  0.133    0.234
jpeg_compression      0.545  0.594  0.671    0.604

## Stage 2 detail — ref_s0 bootstrap CIs

Corruption              Sev   Point  95% CI
----------------------------------------------------
gaussian_noise            1   0.639  [0.613, 0.664]
gaussian_noise            3   0.768  [0.745, 0.791]
gaussian_noise            5   0.948  [0.936, 0.960]
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
random_init              0.998       0.226
mahalanobis              0.998       0.949
mae_untrained              —           —  
mae_trained              0.013       0.147

## Stage 4 — Probe Grid (locked: target mean+zscore, lr-sweep, 200ep)

Model                   n=40   n=200   n=400   n=4000
-----------------------------------------------------
ref_s0                0.3060  0.4260  0.4370   0.5990
ref_s1                0.2660  0.3760  0.4030   0.5640
ref_s2                0.2820  0.3830  0.4270   0.5810
hardmask_s0*          0.3110  0.4320  0.4650   0.5890

---

## Wall-clock summary

Stage 2 (corruption grid):    143s
Stage 3 (OOD):               2792s
Stage 4 (probe grid):          63s
Total:                       2998s (0.83h)