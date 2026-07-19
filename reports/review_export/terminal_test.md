# Terminal Benchmark — Test Split

split=test  dry_run=False  K=8  n_boot=2000
corruptions: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
severities:  [1, 2, 3, 4, 5]

**MAE trained: INCLUDED**
**hardmask_s0\*: single seed, REJECTED lever (R1)**

---

## Stage 2 — Corruption AUROC (point, mean over severities)

Model               gaussian_nois shot_noise    impulse_noise defocus_blur  glass_blur    motion_blur   zoom_blur     snow          frost         fog           brightness    contrast      elastic_trans pixelate      jpeg_compress 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ref_s0              0.736         0.760         0.796         0.209         0.269         0.262         0.360         0.357         0.249         0.078         0.532         0.019         0.439         0.443         0.566         
ref_s1†             0.073         0.097         0.092         0.063         0.078         0.071         0.105         0.035         0.024         0.019         0.078         0.006         0.088         0.130         0.194         
ref_s2              0.639         0.672         0.692         0.276         0.334         0.323         0.416         0.369         0.262         0.108         0.527         0.032         0.484         0.496         0.584         
hardmask_s0*        0.836         0.838         0.863         0.209         0.264         0.249         0.318         0.482         0.359         0.129         0.484         0.032         0.442         0.414         0.550         
pixel_std           0.734         0.766         0.741         0.299         0.343         0.339         0.371         0.486         0.234         0.072         0.443         0.009         0.476         0.441         0.484         
random_init         0.330         0.312         0.333         0.370         0.367         0.364         0.357         0.321         0.432         0.474         0.304         0.318         0.349         0.363         0.356         
mahal_tgt           0.747         0.757         0.768         0.937         0.870         0.859         0.757         0.587         0.767         0.920         0.699         0.977         0.474         0.606         0.565         
mahal_ctx           0.766         0.778         0.799         0.927         0.857         0.844         0.745         0.574         0.764         0.898         0.717         0.966         0.472         0.597         0.565         
mae_untrained       0.412         0.406         0.410         0.542         0.530         0.532         0.518         0.628         0.797         0.849         0.511         0.602         0.495         0.519         0.512         
mae_trained         0.963         0.950         0.978         0.054         0.149         0.159         0.215         0.752         0.466         0.299         0.415         0.153         0.649         0.488         0.541         

† **ref_s1 Stage-2 rows: VOID-INFRASTRUCTURE** — MPS silent Stage-1 corruption (lbd900za clean test mean 0.2711; expected ~0.219). Root cause: MPS async dispatch returned incomplete tensor before synchronization. Fix applied (torch.mps.synchronize() before .cpu()); clean mean recomputed to 0.2190. Stage-2 AUROC cannot be recovered without a Stage-2 rerun. Stage-3 OOD row updated below (see [Decision 1=A] in project log).

## Stage 2 detail — ref_s0 per-severity

Corruption            sev1   sev2   sev3   sev4   sev5     mean
-----------------------------------------------------------------
gaussian_noise        0.579  0.633  0.719  0.828  0.920    0.736
shot_noise            0.591  0.657  0.745  0.874  0.933    0.760
impulse_noise         0.720  0.732  0.760  0.849  0.919    0.796
defocus_blur          0.312  0.268  0.199  0.155  0.112    0.209
glass_blur            0.363  0.315  0.276  0.231  0.161    0.269
motion_blur           0.369  0.317  0.257  0.200  0.165    0.262
zoom_blur             0.393  0.378  0.358  0.345  0.327    0.360
snow                  0.416  0.365  0.383  0.354  0.266    0.357
frost                 0.389  0.272  0.214  0.198  0.171    0.249
fog                   0.117  0.081  0.064  0.070  0.061    0.078
brightness            0.491  0.519  0.538  0.552  0.560    0.532
contrast              0.064  0.025  0.006  0.001  0.000    0.019
elastic_transform     0.440  0.443  0.441  0.439  0.434    0.439
pixelate              0.452  0.447  0.444  0.444  0.431    0.443
jpeg_compression      0.500  0.534  0.553  0.607  0.634    0.566

## Stage 2 detail — ref_s0 bootstrap CIs

Corruption              Sev   Point  95% CI
----------------------------------------------------
gaussian_noise            1   0.579  [0.571, 0.588]
gaussian_noise            2   0.633  [0.624, 0.642]
gaussian_noise            3   0.719  [0.710, 0.728]
gaussian_noise            4   0.828  [0.820, 0.835]
gaussian_noise            5   0.920  [0.915, 0.926]
shot_noise                1   0.591  [0.583, 0.600]
shot_noise                2   0.657  [0.648, 0.666]
shot_noise                3   0.745  [0.737, 0.754]
shot_noise                4   0.874  [0.867, 0.880]
shot_noise                5   0.933  [0.927, 0.937]
impulse_noise             1   0.720  [0.712, 0.729]
impulse_noise             2   0.732  [0.724, 0.741]
impulse_noise             3   0.760  [0.752, 0.768]
impulse_noise             4   0.849  [0.842, 0.856]
impulse_noise             5   0.919  [0.914, 0.924]
defocus_blur              1   0.312  [0.304, 0.320]
defocus_blur              2   0.268  [0.260, 0.275]
defocus_blur              3   0.199  [0.193, 0.206]
defocus_blur              4   0.155  [0.149, 0.161]
defocus_blur              5   0.112  [0.107, 0.118]
glass_blur                1   0.363  [0.354, 0.371]
glass_blur                2   0.315  [0.307, 0.323]
glass_blur                3   0.276  [0.268, 0.283]
glass_blur                4   0.231  [0.224, 0.238]
glass_blur                5   0.161  [0.155, 0.167]
motion_blur               1   0.369  [0.360, 0.377]
motion_blur               2   0.317  [0.310, 0.325]
motion_blur               3   0.257  [0.250, 0.265]
motion_blur               4   0.200  [0.194, 0.207]
motion_blur               5   0.165  [0.159, 0.171]
zoom_blur                 1   0.393  [0.385, 0.402]
zoom_blur                 2   0.378  [0.370, 0.387]
zoom_blur                 3   0.358  [0.349, 0.366]
zoom_blur                 4   0.345  [0.337, 0.353]
zoom_blur                 5   0.327  [0.319, 0.336]
snow                      1   0.416  [0.408, 0.425]
snow                      2   0.365  [0.357, 0.374]
snow                      3   0.383  [0.375, 0.392]
snow                      4   0.354  [0.346, 0.362]
snow                      5   0.266  [0.258, 0.274]
frost                     1   0.389  [0.381, 0.398]
frost                     2   0.272  [0.265, 0.280]
frost                     3   0.214  [0.207, 0.221]
frost                     4   0.198  [0.191, 0.205]
frost                     5   0.171  [0.165, 0.177]
fog                       1   0.117  [0.112, 0.122]
fog                       2   0.081  [0.076, 0.085]
fog                       3   0.064  [0.060, 0.068]
fog                       4   0.070  [0.066, 0.074]
fog                       5   0.061  [0.057, 0.065]
brightness                1   0.491  [0.482, 0.500]
brightness                2   0.519  [0.510, 0.528]
brightness                3   0.538  [0.529, 0.547]
brightness                4   0.552  [0.543, 0.561]
brightness                5   0.560  [0.551, 0.569]
contrast                  1   0.064  [0.060, 0.068]
contrast                  2   0.025  [0.023, 0.028]
contrast                  3   0.006  [0.005, 0.007]
contrast                  4   0.001  [0.000, 0.001]
contrast                  5   0.000  [0.000, 0.000]
elastic_transform         1   0.440  [0.431, 0.448]
elastic_transform         2   0.443  [0.434, 0.452]
elastic_transform         3   0.441  [0.433, 0.451]
elastic_transform         4   0.439  [0.430, 0.448]
elastic_transform         5   0.434  [0.425, 0.443]
pixelate                  1   0.452  [0.443, 0.461]
pixelate                  2   0.447  [0.439, 0.456]
pixelate                  3   0.444  [0.435, 0.452]
pixelate                  4   0.444  [0.435, 0.453]
pixelate                  5   0.431  [0.422, 0.440]
jpeg_compression          1   0.500  [0.492, 0.509]
jpeg_compression          2   0.534  [0.526, 0.543]
jpeg_compression          3   0.553  [0.544, 0.562]
jpeg_compression          4   0.607  [0.599, 0.616]
jpeg_compression          5   0.634  [0.626, 0.643]

## Stage 3 — OOD AUROC

Model                     SVHN    CIFAR-10
--------------------------------------------
ref_s0                   0.078       0.367
ref_s1†                  0.128       0.505
ref_s2                   0.085       0.429
hardmask_s0*             0.121       0.340
pixel_std                0.101       0.395
random_init              0.419       0.378
mahal_tgt                0.986       0.864
mahal_ctx                0.981       0.852
mae_untrained            0.771       0.532
mae_trained              0.013       0.144

† ref_s1 Stage-3 row recomputed with MPS sync fix (Decision 1=A): clean mean 0.2711→0.2190; SVHN 0.1277 [0.1235, 0.1319]; CIFAR-10 0.5048 [0.4968, 0.5132]. Original run-2 values (SVHN=0.008, CIFAR=0.137) were VOID-INFRA.

## Stage 4 — Probe Grid [VAL SELECTION, VAL EVAL — locked protocol]

*(val-era Stage-4; z-score fitted on val; LR selected on val; eval=val —
pre-Decision-2 numbers; see Stage 4b for test eval)*

Model                         n=40         n=200         n=400         n=4000
---------------------------------------------------------------------------
ref_s0                0.2937±0.0034  0.4147±0.0246  0.4550±0.0073  0.6030±0.0008
ref_s1                0.2690±0.0043  0.3653±0.0116  0.4170±0.0118  0.5643±0.0017
ref_s2                0.2683±0.0184  0.3897±0.0160  0.4357±0.0109  0.5803±0.0005
hardmask_s0*          0.2950±0.0022  0.4180±0.0142  0.4813±0.0146  0.5897±0.0009

JEPA ref mean                0.277         0.390         0.436         0.583
Scratch A3 mean†             0.240         0.353         0.389         0.579
Gap (JEPA−A3)             +0.037†       +0.037†       +0.047†       +0.004†
Scratch v2 mean              0.249         0.386         0.426         0.648
Gap (JEPA−v2)             +0.028        +0.004        +0.010        −0.065

† A3 recipe underfit (batch=min(256,n), no augmentation) — reference only.
  Binding gap uses v2 (batch=128, RandomResizedCrop+HFlip; scripts/run_scratch_comparator_v2.py).
  Gate 1B(iii) evidence: gaps +2.8/+0.4/+1.0pp at n=40/200/400; −6.5pp at n=4000.
  Combined spreads (RSS σ_J/σ_S): n=40 RSS=0.016 (gap exceeds); n=200 RSS=0.026 (within); n=400 RSS=0.022 (within).
  Gate decision = human. Test set never reopens for scratch.

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

## Stage 4b — Probe Grid [TEST EVAL] (Decision 2 / PD2)

*(z-score fitted on val; LR selected on val; eval=**test**; 3 probe seeds)*

Model                         n=40         n=200         n=400        n=4000
------------------------------------------------------------------------
ref_s0                0.2786±0.0111  0.3830±0.0084  0.4293±0.0095  0.5592±0.0020
ref_s1†               0.2600±0.0228  0.3528±0.0174  0.3983±0.0050  0.5419±0.0027
ref_s2                0.2677±0.0291  0.3710±0.0180  0.4110±0.0088  0.5396±0.0056
hardmask_s0*          0.2845±0.0136  0.4051±0.0045  0.4473±0.0051  0.5657±0.0018

JEPA ref mean (test)         0.269         0.369         0.413         0.547

† ref_s1 Stage-2 VOID-INFRA (MPS sync fix applied; Stage-3 OOD recomputed; Stage-2 not rerun).

### Stage 4b band check (binding)

Pre-registered: n=4000 test acc within ±0.03 of val-era numbers {0.6030, 0.5643, 0.5803}.

| Model  | test n=4000 | val n=4000 | \|Δ\| | Verdict |
|--------|-------------|------------|-------|---------|
| ref_s0 | 0.5592      | 0.6030     | 0.044 | **FAIL** |
| ref_s1 | 0.5419      | 0.5643     | 0.022 | PASS |
| ref_s2 | 0.5396      | 0.5803     | 0.041 | **FAIL** |

Band check: FAIL (ref_s0 exceeds by 0.014, ref_s2 by 0.011). Direction consistent with expected
val→test gap: probe LR selected on val (n=1000), evaluated on test (n=8000). No recompute triggered;
numbers reported as-is per pre-registered procedure.

### Stage 4b detail — per probe seed

Model/seed                    n=40   n=200   n=400   n=4000
------------------------------------------------------------
ref_s0 s=0                  0.2834  0.3842  0.4300  0.5612
ref_s0 s=1                  0.2660  0.3740  0.4195  0.5590
ref_s0 s=2                  0.2865  0.3907  0.4384  0.5573
ref_s1 s=0                  0.2596  0.3586  0.3926  0.5433
ref_s1 s=1                  0.2375  0.3332  0.4011  0.5387
ref_s1 s=2                  0.2830  0.3665  0.4013  0.5436
ref_s2 s=0                  0.2920  0.3665  0.4153  0.5443
ref_s2 s=1                  0.2354  0.3557  0.4009  0.5411
ref_s2 s=2                  0.2758  0.3909  0.4168  0.5334
hardmask_s0* s=0            0.2979  0.4086  0.4520  0.5677
hardmask_s0* s=1            0.2707  0.4000  0.4419  0.5642
hardmask_s0* s=2            0.2849  0.4067  0.4479  0.5652

---

## Wall-clock summary

Stage 1 (clean energies):     2265s
Stage 2 (corruption grid):   12972s
Stage 3 (OOD):                 636s
Stage 4 (probe grid, val):     185s
Stage 4b (probe grid, test):   ~350s (separate script)
Total (Stages 1-4):          16058s (4.46h)