# Terminal Benchmark — Val Split

split=val  dry_run=False  K=8  n_boot=2000
corruptions: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
severities:  [1, 2, 3, 4, 5]

**MAE trained: INCLUDED**
**hardmask_s0\*: single seed, REJECTED lever (R1)**
**SKIPPED (library incompatibility): glass_blur/sev1, glass_blur/sev2, glass_blur/sev3, glass_blur/sev4, glass_blur/sev5, fog/sev1, fog/sev2, fog/sev3, fog/sev4, fog/sev5**
  glass_blur: scikit-image gaussian() multichannel kwarg removed
  fog:        numpy np.float_ removed in NumPy 2.0

---

## Stage 2 — Corruption AUROC (point, mean over severities)

Model               gaussian_nois shot_noise    impulse_noise defocus_blur  glass_blur    motion_blur   zoom_blur     snow          frost         fog           brightness    contrast      elastic_trans pixelate      jpeg_compress 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ref_s0              0.781         0.802         0.836         0.236           —           0.293         0.389         0.408         0.284           —           0.578         0.020         0.483         0.484         0.607         
ref_s1              0.661         0.689         0.713         0.336           —           0.372         0.452         0.325         0.249           —           0.504         0.033         0.484         0.532         0.630         
ref_s2              0.646         0.677         0.695         0.279           —           0.325         0.415         0.378         0.261           —           0.532         0.029         0.488         0.497         0.587         
hardmask_s0*        0.878         0.880         0.900         0.258           —           0.303         0.370         0.551         0.421           —           0.546         0.044         0.498         0.470         0.606         
pixel_std           0.738         0.769         0.744         0.300           —           0.338         0.379         0.501         0.241           —           0.459         0.009         0.476         0.442         0.484         
random_init         0.456         0.541         0.189         0.151           —           0.252         0.261         0.270         0.639           —           0.145         0.304         0.365         0.235         0.220         
mahalanobis         0.863         0.868         0.892         0.977           —           0.917         0.837         0.719         0.872           —           0.806         0.991         0.506         0.640         0.662         
mae_untrained       0.417         0.440         0.401         0.534           —           0.524         0.518         0.298         0.285           —           0.308         0.591         0.495         0.509         0.504         
mae_trained         0.966         0.954         0.979         0.050           —           0.149         0.207         0.766         0.470           —           0.415         0.143         0.653         0.486         0.541         

## Stage 2 detail — ref_s0 per-severity

Corruption            sev1   sev2   sev3   sev4   sev5     mean
-----------------------------------------------------------------
gaussian_noise        0.635  0.687  0.766  0.867  0.948    0.781
shot_noise            0.650  0.711  0.790  0.906  0.956    0.802
impulse_noise         0.769  0.776  0.803  0.885  0.946    0.836
defocus_blur          0.341  0.297  0.228  0.181  0.133    0.236
glass_blur              —      —      —      —      —        —  
motion_blur           0.405  0.351  0.287  0.234  0.188    0.293
zoom_blur             0.426  0.408  0.385  0.372  0.353    0.389
snow                  0.465  0.415  0.437  0.409  0.317    0.408
frost                 0.436  0.311  0.249  0.227  0.198    0.284
fog                     —      —      —      —      —        —  
brightness            0.534  0.564  0.584  0.600  0.609    0.578
contrast              0.068  0.024  0.005  0.001  0.000    0.020
elastic_transform     0.482  0.486  0.484  0.484  0.480    0.483
pixelate              0.493  0.488  0.486  0.485  0.466    0.484
jpeg_compression      0.545  0.580  0.594  0.646  0.671    0.607

## Stage 2 detail — ref_s0 bootstrap CIs

Corruption              Sev   Point  95% CI
----------------------------------------------------
gaussian_noise            1   0.635  [0.610, 0.660]
gaussian_noise            2   0.687  [0.662, 0.711]
gaussian_noise            3   0.766  [0.742, 0.789]
gaussian_noise            4   0.867  [0.848, 0.885]
gaussian_noise            5   0.948  [0.935, 0.960]
shot_noise                1   0.650  [0.625, 0.674]
shot_noise                2   0.711  [0.687, 0.735]
shot_noise                3   0.790  [0.768, 0.812]
shot_noise                4   0.906  [0.891, 0.922]
shot_noise                5   0.956  [0.945, 0.966]
impulse_noise             1   0.769  [0.747, 0.791]
impulse_noise             2   0.776  [0.754, 0.797]
impulse_noise             3   0.803  [0.782, 0.824]
impulse_noise             4   0.885  [0.867, 0.902]
impulse_noise             5   0.946  [0.934, 0.959]
defocus_blur              1   0.341  [0.317, 0.366]
defocus_blur              2   0.297  [0.274, 0.320]
defocus_blur              3   0.228  [0.208, 0.249]
defocus_blur              4   0.181  [0.163, 0.199]
defocus_blur              5   0.133  [0.118, 0.149]
motion_blur               1   0.405  [0.379, 0.430]
motion_blur               2   0.351  [0.326, 0.375]
motion_blur               3   0.287  [0.264, 0.310]
motion_blur               4   0.234  [0.214, 0.255]
motion_blur               5   0.188  [0.169, 0.207]
zoom_blur                 1   0.426  [0.399, 0.451]
zoom_blur                 2   0.408  [0.382, 0.433]
zoom_blur                 3   0.385  [0.360, 0.410]
zoom_blur                 4   0.372  [0.346, 0.396]
zoom_blur                 5   0.353  [0.328, 0.378]
snow                      1   0.465  [0.439, 0.490]
snow                      2   0.415  [0.390, 0.442]
snow                      3   0.437  [0.411, 0.463]
snow                      4   0.409  [0.383, 0.435]
snow                      5   0.317  [0.294, 0.341]
frost                     1   0.436  [0.410, 0.462]
frost                     2   0.311  [0.288, 0.336]
frost                     3   0.249  [0.227, 0.270]
frost                     4   0.227  [0.205, 0.248]
frost                     5   0.198  [0.178, 0.219]
brightness                1   0.534  [0.509, 0.560]
brightness                2   0.564  [0.538, 0.589]
brightness                3   0.584  [0.559, 0.609]
brightness                4   0.600  [0.575, 0.624]
brightness                5   0.609  [0.584, 0.634]
contrast                  1   0.068  [0.058, 0.080]
contrast                  2   0.024  [0.018, 0.030]
contrast                  3   0.005  [0.003, 0.007]
contrast                  4   0.001  [0.000, 0.001]
contrast                  5   0.000  [0.000, 0.001]
elastic_transform         1   0.482  [0.456, 0.508]
elastic_transform         2   0.486  [0.461, 0.512]
elastic_transform         3   0.484  [0.459, 0.511]
elastic_transform         4   0.484  [0.459, 0.511]
elastic_transform         5   0.480  [0.455, 0.506]
pixelate                  1   0.493  [0.467, 0.520]
pixelate                  2   0.488  [0.462, 0.514]
pixelate                  3   0.486  [0.460, 0.512]
pixelate                  4   0.485  [0.459, 0.511]
pixelate                  5   0.466  [0.439, 0.491]
jpeg_compression          1   0.545  [0.519, 0.571]
jpeg_compression          2   0.580  [0.554, 0.606]
jpeg_compression          3   0.594  [0.569, 0.620]
jpeg_compression          4   0.646  [0.622, 0.670]
jpeg_compression          5   0.671  [0.648, 0.695]

## Stage 3 — OOD AUROC

Model                     SVHN    CIFAR-10
--------------------------------------------
ref_s0                   0.098       0.411
ref_s1                   0.133       0.518
ref_s2                   0.088       0.438
hardmask_s0*             0.165       0.401
pixel_std                0.102       0.405
random_init              0.680       0.194
mahalanobis              0.998       0.949
mae_untrained              —           —  
mae_trained              0.013       0.147

## Stage 4 — Probe Grid (locked: target mean+zscore, lr-sweep, 200ep, 3 probe seeds)

Model                         n=40         n=200         n=400         n=4000
---------------------------------------------------------------------------
ref_s0                0.2823±0.0037  0.4163±0.0187  0.4500±0.0059  0.6027±0.0005
ref_s1                0.2713±0.0063  0.3673±0.0131  0.4133±0.0147  0.5643±0.0021
ref_s2                0.2683±0.0268  0.3927±0.0155  0.4367±0.0054  0.5810±0.0008
hardmask_s0*          0.2953±0.0045  0.4267±0.0093  0.4833±0.0125  0.5850±0.0000

### Stage 4 detail — per probe seed

Model/seed                    n=40   n=200   n=400   n=4000
------------------------------------------------------------
ref_s0 s=0                  0.2870  0.4270  0.4420  0.6030
ref_s0 s=1                  0.2820  0.3900  0.4520  0.6030
ref_s0 s=2                  0.2780  0.4320  0.4560  0.6020
ref_s1 s=0                  0.2800  0.3790  0.3960  0.5670
ref_s1 s=1                  0.2650  0.3490  0.4120  0.5620
ref_s1 s=2                  0.2690  0.3740  0.4320  0.5640
ref_s2 s=0                  0.2980  0.3940  0.4290  0.5800
ref_s2 s=1                  0.2330  0.3730  0.4400  0.5810
ref_s2 s=2                  0.2740  0.4110  0.4410  0.5820
hardmask_s0* s=0            0.3010  0.4300  0.4800  0.5850
hardmask_s0* s=1            0.2900  0.4140  0.4700  0.5850
hardmask_s0* s=2            0.2950  0.4360  0.5000  0.5850

---

## Wall-clock summary

Stage 2 (corruption grid):   1813s
Stage 3 (OOD):                795s
Stage 4 (probe grid):         278s
Total:                       2887s (0.80h)