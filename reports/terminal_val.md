# Terminal Benchmark — Val Split

split=val  dry_run=False  K=8  n_boot=2000
corruptions: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
severities:  [1, 2, 3, 4, 5]

**MAE trained: INCLUDED**
**hardmask_s0\*: single seed, REJECTED lever (R1)**

---

## Stage 2 — Corruption AUROC (point, mean over severities)

Model               gaussian_nois shot_noise    impulse_noise defocus_blur  glass_blur    motion_blur   zoom_blur     snow          frost         fog           brightness    contrast      elastic_trans pixelate      jpeg_compress 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ref_s0              0.780         0.803         0.835         0.236         0.299         0.292         0.389         0.407         0.285         0.085         0.578         0.020         0.484         0.484         0.607         
ref_s1              0.660         0.689         0.712         0.336         0.392         0.372         0.452         0.325         0.248         0.136         0.504         0.033         0.483         0.532         0.630         
ref_s2              0.646         0.678         0.694         0.279         0.339         0.325         0.415         0.376         0.261         0.103         0.532         0.029         0.488         0.497         0.587         
hardmask_s0*        0.878         0.879         0.900         0.258         0.317         0.302         0.370         0.550         0.417         0.167         0.546         0.044         0.497         0.470         0.606         
pixel_std           0.738         0.769         0.744         0.300         0.344         0.338         0.379         0.501         0.240         0.069         0.459         0.009         0.477         0.442         0.484         
random_init         0.132         0.245         0.138         0.165         0.047         0.107         0.114         0.078         0.194         0.151         0.046         0.319         0.217         0.213         0.074         
mahalanobis         0.863         0.870         0.892         0.977         0.935         0.919         0.837         0.719         0.870         0.971         0.806         0.991         0.507         0.640         0.662         
mae_untrained       0.381         0.384         0.374         0.552         0.537         0.538         0.517         0.613         0.802         0.737         0.515         0.609         0.495         0.519         0.512         
mae_trained         0.966         0.954         0.979         0.050         0.141         0.149         0.207         0.766         0.470         0.290         0.415         0.143         0.653         0.486         0.541         

## Stage 2 detail — ref_s0 per-severity

Corruption            sev1   sev2   sev3   sev4   sev5     mean
-----------------------------------------------------------------
gaussian_noise        0.635  0.685  0.765  0.866  0.948    0.780
shot_noise            0.649  0.714  0.792  0.906  0.956    0.803
impulse_noise         0.762  0.778  0.803  0.883  0.946    0.835
defocus_blur          0.341  0.297  0.228  0.181  0.133    0.236
glass_blur            0.397  0.345  0.305  0.259  0.187    0.299
motion_blur           0.405  0.349  0.283  0.229  0.196    0.292
zoom_blur             0.426  0.408  0.385  0.372  0.353    0.389
snow                  0.466  0.417  0.437  0.404  0.313    0.407
frost                 0.438  0.306  0.254  0.226  0.201    0.285
fog                   0.131  0.088  0.069  0.073  0.065    0.085
brightness            0.534  0.564  0.584  0.600  0.609    0.578
contrast              0.068  0.024  0.005  0.001  0.000    0.020
elastic_transform     0.486  0.485  0.484  0.483  0.481    0.484
pixelate              0.493  0.488  0.486  0.485  0.466    0.484
jpeg_compression      0.545  0.580  0.594  0.646  0.671    0.607

## Stage 2 detail — ref_s0 bootstrap CIs

Corruption              Sev   Point  95% CI
----------------------------------------------------
gaussian_noise            1   0.635  [0.611, 0.661]
gaussian_noise            2   0.685  [0.660, 0.710]
gaussian_noise            3   0.765  [0.742, 0.788]
gaussian_noise            4   0.866  [0.847, 0.885]
gaussian_noise            5   0.948  [0.935, 0.960]
shot_noise                1   0.649  [0.624, 0.674]
shot_noise                2   0.714  [0.689, 0.738]
shot_noise                3   0.792  [0.771, 0.814]
shot_noise                4   0.906  [0.890, 0.921]
shot_noise                5   0.956  [0.945, 0.967]
impulse_noise             1   0.762  [0.741, 0.784]
impulse_noise             2   0.778  [0.756, 0.800]
impulse_noise             3   0.803  [0.782, 0.825]
impulse_noise             4   0.883  [0.866, 0.900]
impulse_noise             5   0.946  [0.934, 0.958]
defocus_blur              1   0.341  [0.317, 0.366]
defocus_blur              2   0.297  [0.274, 0.320]
defocus_blur              3   0.228  [0.208, 0.249]
defocus_blur              4   0.181  [0.163, 0.199]
defocus_blur              5   0.133  [0.118, 0.149]
glass_blur                1   0.397  [0.371, 0.421]
glass_blur                2   0.345  [0.321, 0.370]
glass_blur                3   0.305  [0.282, 0.328]
glass_blur                4   0.259  [0.238, 0.281]
glass_blur                5   0.187  [0.168, 0.205]
motion_blur               1   0.405  [0.380, 0.431]
motion_blur               2   0.349  [0.324, 0.374]
motion_blur               3   0.283  [0.260, 0.306]
motion_blur               4   0.229  [0.208, 0.249]
motion_blur               5   0.196  [0.177, 0.215]
zoom_blur                 1   0.426  [0.399, 0.451]
zoom_blur                 2   0.408  [0.382, 0.433]
zoom_blur                 3   0.385  [0.360, 0.410]
zoom_blur                 4   0.372  [0.346, 0.396]
zoom_blur                 5   0.353  [0.328, 0.378]
snow                      1   0.466  [0.440, 0.492]
snow                      2   0.417  [0.392, 0.444]
snow                      3   0.437  [0.411, 0.463]
snow                      4   0.404  [0.378, 0.430]
snow                      5   0.313  [0.290, 0.338]
frost                     1   0.438  [0.413, 0.465]
frost                     2   0.306  [0.282, 0.330]
frost                     3   0.254  [0.232, 0.278]
frost                     4   0.226  [0.205, 0.247]
frost                     5   0.201  [0.181, 0.222]
fog                       1   0.131  [0.115, 0.147]
fog                       2   0.088  [0.075, 0.101]
fog                       3   0.069  [0.058, 0.080]
fog                       4   0.073  [0.062, 0.085]
fog                       5   0.065  [0.054, 0.076]
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
elastic_transform         1   0.486  [0.461, 0.512]
elastic_transform         2   0.485  [0.459, 0.510]
elastic_transform         3   0.484  [0.458, 0.510]
elastic_transform         4   0.483  [0.457, 0.509]
elastic_transform         5   0.481  [0.456, 0.507]
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
random_init              0.118       0.028
mahalanobis              0.998       0.949
mae_untrained              —           —  
mae_trained              0.013       0.147

## Stage 4 — Probe Grid (locked: target mean+zscore, lr-sweep, 200ep, 3 probe seeds)

Model                         n=40         n=200         n=400         n=4000
---------------------------------------------------------------------------
ref_s0                0.2920±0.0067  0.4177±0.0177  0.4563±0.0088  0.6003±0.0005
ref_s1                0.2690±0.0033  0.3647±0.0058  0.4147±0.0160  0.5650±0.0022
ref_s2                0.2740±0.0221  0.3913±0.0184  0.4367±0.0068  0.5793±0.0017
hardmask_s0*          0.3003±0.0090  0.4187±0.0111  0.4823±0.0119  0.5870±0.0036

### Stage 4 detail — per probe seed

Model/seed                    n=40   n=200   n=400   n=4000
------------------------------------------------------------
ref_s0 s=0                  0.2990  0.4260  0.4440  0.6010
ref_s0 s=1                  0.2940  0.3930  0.4640  0.6000
ref_s0 s=2                  0.2830  0.4340  0.4610  0.6000
ref_s1 s=0                  0.2690  0.3660  0.3940  0.5680
ref_s1 s=1                  0.2730  0.3570  0.4170  0.5630
ref_s1 s=2                  0.2650  0.3710  0.4330  0.5640
ref_s2 s=0                  0.3020  0.3990  0.4270  0.5810
ref_s2 s=1                  0.2480  0.3660  0.4410  0.5800
ref_s2 s=2                  0.2720  0.4090  0.4420  0.5770
hardmask_s0* s=0            0.3010  0.4260  0.4760  0.5850
hardmask_s0* s=1            0.3110  0.4030  0.4720  0.5840
hardmask_s0* s=2            0.2890  0.4270  0.4990  0.5920

---

## Wall-clock summary

Stage 2 (corruption grid):   2626s
Stage 3 (OOD):                578s
Stage 4 (probe grid):         185s
Total:                       3389s (0.94h)