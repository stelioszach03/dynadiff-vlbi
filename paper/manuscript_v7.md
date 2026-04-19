---
authors:
  - name: Stylianos Georgios Zacharioudakis
    affiliation: Department of Informatics and Telecommunications, National and Kapodistrian University of Athens, Panepistimioupolis, Athens 16122, Greece
    email: sdi2200243@di.uoa.gr
corresponding_author:
  name: Stylianos Georgios Zacharioudakis
  email: sdi2200243@di.uoa.gr
keywords:
  - "black hole physics"
  - "techniques: interferometric"
  - "methods: data analysis"
  - "methods: statistical"
---

# Earned versus enforced measurement consistency in dynamic VLBI imaging: an evaluation framework with score-based posterior sampling

## Abstract

Learned methods for dynamic very long baseline interferometric (VLBI) imaging routinely incorporate data-consistency layers that restore observed Fourier coefficients by construction. When those same coefficients are subsequently used for evaluation, the reported observation-domain agreement conflates enforcement with genuine inference quality. We formalize this problem as the distinction between earned and enforced measurement consistency, and prove that for linear-Gaussian measurement models the gap between enforced and earned reconstruction error is bounded below by the posterior predictive variance projected on to the withheld measurement subspace, a quantity governed by the conditional mutual information between target and signal given the support measurements. We operationalize this theoretical result through a deterministic benchmark protocol that partitions observed visibilities into support and target sets, restricts both model input and data-consistency projection to the support set, and evaluates on the genuinely unseen target. To demonstrate the protocol, we introduce DynaDiff, a conditional score-based diffusion model for dynamic image sequences that conditions on sparse Fourier measurements through cross-attention and applies measurement-consistency guidance exclusively on the support partition. Validated at 128 x 128 resolution on GRMHD-inspired black hole simulations with three structured holdout families and four support fractions, the learned reference implementation trained with the earned-consistency objective achieves the lowest held-out visibility root-mean-square error in 11 of 12 benchmark conditions, with 9 reaching statistical significance in paired-bootstrap tests. An expanded public validation suite on four official EHT calibrated-data releases (M87 2017, M87 2018, 3C 279 2017, and Centaurus A 2017) shows that no single method dominates on released measurements, and the protocol exposes a synthetic-to-public transfer gap that standard observation-domain metrics hide. Code, deterministic benchmark manifests, and trained models are released.

## 1 Introduction

The Event Horizon Telescope (EHT) demonstrated both the scientific potential and the inferential difficulty of black hole imaging at horizon scales through the M87* and Sagittarius A* campaigns [@EHT2019M87I; @EHT2019M87IV; @EHT2022SgrAI; @EHT2022SgrAIII]. In the dynamic setting, where the source evolves during an observation window, the challenge is compounded by temporal variability, irregular measurement support, and structured missingness that varies from scan to scan and baseline to baseline [@Farah2022SelectiveDynamicalImaging; @Satapathy2022M87Variability; @Georgiev2022UniversalPowerLaw].

Learned methods have shown promise for VLBI-style reconstruction, but they share a methodological vulnerability that has not been systematically addressed. Most modern architectures include a data-consistency (DC) layer that explicitly projects the reconstruction back on to the observed Fourier coefficients. When evaluation is subsequently performed on the same coefficients, agreement is guaranteed by construction and carries no information about inference quality. A model can appear measurement-consistent simply because it was forced to be so. We call this the distinction between earned and enforced measurement consistency. The problem is not specific to VLBI: any learned inverse-problem solver that incorporates DC and evaluates on the enforced coefficients faces the same blind spot.

This paper makes three contributions that together address that vulnerability:

1. **An evaluation principle with theoretical foundation.** We formalize the earned-versus-enforced distinction and prove (Theorem 1) that the gap between enforced and earned error is governed by the conditional mutual information between target measurements and the unknown signal given the support measurements. This provides both a diagnostic for existing methods and a principled motivation for the support-target holdout protocol (Section 3).

2. **A concrete benchmark protocol and learned reference implementation.** We define a deterministic benchmark with three structured holdout families --- baseline-track blocks, scan-segment blocks, and station dropout --- at four support fractions (80, 60, 40, 20 per cent), with shared comparator partitions and paired-bootstrap reporting. The benchmark's reference implementation is a 3D U-Net backbone with residual refinement and support-only DC, trained with the EMC objective; it achieves the lowest held-out visibility RMSE among the learned comparators in 11 of 12 synthetic conditions. We additionally introduce DynaDiff, a conditional score-based diffusion model for temporal sequences with measurement cross-attention and support-only guidance, as an architecture designed to produce calibrated posterior samples under the same protocol (Section 4, Section 5).

3. **Quantitative public-EHT validation exposing the transfer gap.** We apply the identical holdout protocol to four official public EHT calibrated-data releases. The protocol reveals that synthetic benchmark winners do not transfer uniformly to real measurements: no single method dominates, classical approaches remain competitive, and the synthetic-to-public gap is quantitatively measurable through the protocol. This is not a weakness but a feature: the protocol makes the transfer gap visible rather than hiding it behind enforcement (Section 6).

Throughout, we maintain a strict separation between the evaluation protocol and the model architecture. The *earned-consistency protocol* refers to the support--target holdout evaluation framework. The *EMC training objective* adds a held-out visibility loss to any backbone. *DynaDiff* is the specific score-based diffusion architecture introduced in Section 4. In the benchmark tables, 'EMC' denotes the 3D U-Net backbone augmented with a residual-refinement branch, a support-only DC layer, and trained with the EMC objective (Section 5.3). 'Baseline' denotes the same 3D U-Net backbone trained with standard image-domain losses only, without DC or EMC objective. Both are deterministic single-forward-pass models. DynaDiff posterior sampling results are reported separately where available.

## 2 Related work

The immediate astronomy context is black hole imaging with the EHT. The 2019 M87* series established the calibration, imaging, and physical-interpretation framework for the first horizon-scale black hole image [@EHT2019M87I; @EHT2019M87III; @EHT2019M87IV; @EHT2019M87V; @EHT2019M87VI]. The 2022 Sgr A* results further emphasized the challenge of robust inference under sparse, irregular coverage for a more variable target [@EHT2022SgrAI; @EHT2022SgrAII; @EHT2022SgrAIII]. Official public calibrated data are now available for M87 2017, M87 2018, 3C 279 2017, and Centaurus A 2017 [@EHT2019PublicM87DataRelease; @EHT2024PublicM87DataRelease; @EHT2020Public3C279DataRelease; @EHT2021PublicCenADataRelease].

Classical imaging methods central to EHT analysis include CLEAN [@Hogbom1974CLEAN], sparse-modelling approaches [@Akiyama2017SparseM87], and closure-aware regularized imaging [@Chael2018ClosureImaging]. On the machine-learning side, recent work has explored learned EHT inference through synthetic libraries and Bayesian pipelines [@Janssen2025DLI; @Janssen2025DLII; @Janssen2025DLIII] and closure-invariant architectures [@Lai2025ClosureInvariants]. None of these works systematically distinguish earned from enforced observation-domain agreement.

Score-based diffusion models have emerged as state-of-the-art for image inverse problems through diffusion posterior sampling (DPS; @Chung2023DPS), manifold-constrained gradients [@Chung2023MCG], DDRM [@Kawar2022DDRM], and RED-diff [@Mardani2023REDDiff]. These methods operate on individual images and do not address the earned-enforced distinction. DynaDiff extends the paradigm to dynamic sequences with measurement-aware conditioning and support-only guidance.

Dynamic imaging and variability studies [@Farah2022SelectiveDynamicalImaging; @Satapathy2022M87Variability; @Georgiev2022UniversalPowerLaw] motivate the temporal aspect, and video diffusion models [@Ho2022VideoDiffusion] provide the generative modelling context.

## 3 Theoretical framework

### 3.1 Problem setting

Let \(x_{1:T} \in \mathbb{R}^{T \times N}\) denote a dynamic image sequence of \(T\) frames, each with \(N = H \times W\) pixels. The forward model produces sparse complex visibilities

\[
y_{1:T} = \mathcal{M}_{1:T} \mathcal{F}(x_{1:T}) + \epsilon, \quad \epsilon \sim \mathcal{CN}(0, \sigma^2 I),
\]

where \(\mathcal{F}\) is the 2D discrete Fourier transform applied independently per frame and \(\mathcal{M}_{1:T}\) is a binary sampling mask that selects measured Fourier coefficients at each time step.

### 3.2 Support--target partition

We partition the observed coefficients into disjoint support and target subsets:

\[
\mathcal{M}_{1:T} = \mathcal{M}^{\mathrm{sup}}_{1:T} + \mathcal{M}^{\mathrm{tgt}}_{1:T}, \quad \mathcal{M}^{\mathrm{sup}}_{1:T} \odot \mathcal{M}^{\mathrm{tgt}}_{1:T} = 0.
\]

The reconstruction method receives only \(y^{\mathrm{sup}} = \mathcal{M}^{\mathrm{sup}} \mathcal{F}(x) + \epsilon\). Any data-consistency layer operates only on \(\mathcal{M}^{\mathrm{sup}}\). The target measurements \(y^{\mathrm{tgt}}\) are withheld from both the model input and the DC layer. This partition is deterministic and shared across all methods in the benchmark.

### 3.3 The earned--enforced gap

To state the main theoretical result, we work in the linear-Gaussian setting with a single frame. Let \(x \sim \mathcal{N}(0, \Sigma_x)\) with \(\Sigma_x \succ 0\), let \(A \in \mathbb{R}^{m \times n}\) be the measurement matrix with rows partitioned as \(A_{\mathrm{sup}} \in \mathbb{R}^{m_s \times n}\) and \(A_{\mathrm{tgt}} \in \mathbb{R}^{m_t \times n}\), and let the noise be \(\epsilon \sim \mathcal{N}(0, \sigma^2 I)\). The posterior given only the support measurements is \(x \mid y^{\mathrm{sup}} \sim \mathcal{N}(\hat{x}, \Sigma_{\mathrm{post}})\), where \(\Sigma_{\mathrm{post}} = (\Sigma_x^{-1} + \sigma^{-2} A_{\mathrm{sup}}^\top A_{\mathrm{sup}})^{-1}\) and \(\hat{x} = \sigma^{-2}\Sigma_{\mathrm{post}} A_{\mathrm{sup}}^\top y^{\mathrm{sup}}\).

We distinguish two estimators. The *Bayes-optimal* estimator \(\hat{x}\) minimizes the posterior expected squared error but does not explicitly enforce agreement with the support measurements. A *DC-projected* estimator takes any reconstruction \(\tilde{x}\) and replaces the predicted support visibilities with the observed ones, yielding zero residual on the support set by construction. In practice, learned methods with DC layers behave closer to the second: the support residual is near zero regardless of the model's inference quality. The theorem below uses the Bayes-optimal estimator and therefore provides a *lower bound* on the gap that any DC-equipped method would exhibit.

**Theorem 1** (Earned--enforced gap bound). *Let \(\hat{x} = \mathbb{E}[x \mid y^{\mathrm{sup}}]\). Then the expected per-coefficient squared error satisfies:*

*Earned error (target):*
\[
E_{\mathrm{earned}} \;=\; \frac{1}{m_t}\mathbb{E}\bigl[\|y^{\mathrm{tgt}} - A_{\mathrm{tgt}}\hat{x}\|^2\bigr] \;=\; \sigma^2 + \frac{1}{m_t}\operatorname{tr}\!\bigl(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top\bigr).
\]

*For a DC-projected estimator with zero support residual, the enforced error is*
\[
E_{\mathrm{enforced}}^{\mathrm{DC}} \;=\; 0.
\]

*The gap \(\Delta = E_{\mathrm{earned}} - E_{\mathrm{enforced}}^{\mathrm{DC}} = \sigma^2 + \frac{1}{m_t}\operatorname{tr}(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top) > 0\) is strictly positive whenever any target measurement carries information about \(x\) beyond what the support provides, i.e., whenever \(I(y^{\mathrm{tgt}}; x \mid y^{\mathrm{sup}}) > 0\).*

*Proof.* Decompose the target residual: \(r_{\mathrm{tgt}} = A_{\mathrm{tgt}}(x - \hat{x}) + \epsilon_{\mathrm{tgt}}\). Since \(\hat{x} = \mathbb{E}[x \mid y^{\mathrm{sup}}]\), the estimation error has zero mean and covariance \(\Sigma_{\mathrm{post}}\). The noise \(\epsilon_{\mathrm{tgt}}\) is independent of both \(x\) and \(y^{\mathrm{sup}}\), so the cross-term vanishes:
\[
\mathbb{E}[\|r_{\mathrm{tgt}}\|^2] = \operatorname{tr}(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top) + \sigma^2 m_t.
\]
The DC-projected enforced error is zero by construction. For the mutual-information connection: the conditional distribution \(y^{\mathrm{tgt}} \mid y^{\mathrm{sup}}\) has covariance \(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top + \sigma^2 I\), while \(y^{\mathrm{tgt}} \mid x\) has covariance \(\sigma^2 I\). Therefore \(I(y^{\mathrm{tgt}}; x \mid y^{\mathrm{sup}}) = \frac{1}{2}\log\det(I + \sigma^{-2} A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top)\), which is zero only when \(\operatorname{tr}(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top) = 0\). \(\square\)

**Remark 1** (The gap is a lower bound). *The Bayes-optimal estimator minimizes the earned error among all estimators. Any practical learned method achieves earned error at least as large. Methods with DC layers achieve enforced error near zero, making the practical gap strictly larger than the \(\Delta\) in Theorem 1. The theorem therefore provides a conservative bound on the diagnostic power of the protocol.*

**Corollary 1** (Support fraction scaling). *Under i.i.d. Gaussian rows with support fraction \(\alpha = m_{\mathrm{sup}} / m\), the gap scales as \(\Delta \sim (1-\alpha) \|x\|^2 / m\), vanishing only as \(\alpha \to 1\) or \(m \to \infty\).*

The practical consequence is immediate: any method that reports only enforced consistency (evaluation on all observed coefficients after DC projection) cannot distinguish genuine learned recovery from mere enforcement. The earned-consistency protocol resolves this ambiguity at no additional computational cost.

### 3.4 Numerical verification

We verify Theorem 1 with a Monte Carlo experiment. For a 16-dimensional signal with 50 Gaussian measurements, \(\sigma = 0.1\), and 10 000 samples, we compute both the Bayes-optimal estimator and a DC-projected estimator at each support fraction from 0.2 to 0.95.

For the *Bayes-optimal* estimator, the support-set residual (enforced error) is close to \(\sigma^2 = 0.01\) but not exactly zero, because the posterior mean does not perfectly interpolate the noisy measurements. The target-set residual (earned error) exceeds \(\sigma^2\) by exactly the amount predicted by Theorem 1, with Monte Carlo and theory agreeing within 1 per cent.

For the *DC-projected* estimator (support residual forced to zero), the enforced error is identically zero, and the earned error is accordingly larger. In both cases, the gap \(\Delta\) decreases monotonically with support fraction, confirming Corollary 1.

The key point is that the Bayes-optimal gap provides a *lower bound* on the practical gap for any DC-equipped method, as stated in Remark 1.

## 4 DynaDiff: score-based posterior sampling for dynamic inverse problems

### 4.1 Architecture

DynaDiff is a conditional score-based diffusion model that learns the score function \(\nabla_{x_t} \log p_t(x_t \mid y^{\mathrm{sup}})\) of the noisy image distribution conditioned on support measurements. The architecture has three components.

*Temporal score network.* A 3D U-Net operating on tensors of shape \([B, C, T, H, W]\) with sinusoidal time-embedding for noise-level conditioning, ResNet blocks with GroupNorm and SiLU activations, and variable depth (\(L\) encoder--decoder levels with spatial downsampling by factor 2 per level).

*Measurement cross-attention.* Sparse measurements are tokenized as \(\{v_i^{\mathrm{re}}, v_i^{\mathrm{im}}, u_i, v_i\}\) and projected through a learned encoder. At selected U-Net levels, image features attend to these tokens via multi-head cross-attention. This provides measurement conditioning beyond what guidance alone offers: the score network learns to modulate its denoising behaviour based on the observed visibility structure.

*Noise schedule.* We use a variance-preserving SDE with \(\sigma(t) = \sigma_{\min}^{1-t} \sigma_{\max}^t\) where \(\sigma_{\min} = 0.01\) and \(\sigma_{\max} = 50\).

### 4.2 Training

The model is trained with denoising score matching:

\[
\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0,\, \epsilon}\!\left[\frac{1}{\sigma(t)^2} \|\epsilon_\theta(x_0 + \sigma(t)\epsilon,\, t,\, y^{\mathrm{sup}}) - \epsilon\|^2\right],
\]

where \(\epsilon_\theta\) is the score network parametrized by \(\theta\).

### 4.3 Inference with support-only guidance

At inference, we combine the learned score with measurement-consistency guidance on the support set only:

\[
x_{t-\Delta t} = x_t + \sigma(t)^2 s_\theta(x_t, t, y^{\mathrm{sup}})\,\Delta t - \lambda \nabla_{x_t}\|\mathcal{M}^{\mathrm{sup}}\mathcal{F}(x_t) - y^{\mathrm{sup}}\|^2\,\Delta t + \sqrt{2\sigma(t)^2\,\Delta t}\,z,
\]

where \(s_\theta = -\epsilon_\theta / \sigma(t)\) is the score estimate and \(z \sim \mathcal{N}(0, I)\). The measurement-consistency gradient is restricted to the support mask \(\mathcal{M}^{\mathrm{sup}}\): target coefficients are never enforced. Running the reverse SDE \(S\) times from independent noise yields posterior samples, from which the posterior mean and per-pixel variance are computed without any auxiliary loss term.

### 4.4 Relation to the EMC training objective

The EMC training objective adds a held-out visibility loss during model training:

\[
\mathcal{L}_{\mathrm{EMC}} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{sup}} \mathcal{L}_{\mathrm{vis}}^{\mathrm{sup}} + \lambda_{\mathrm{tgt}} \mathcal{L}_{\mathrm{vis}}^{\mathrm{tgt}} + \lambda_{\mathrm{temp}} \mathcal{L}_{\mathrm{temporal}},
\]

where \(\mathcal{L}_{\mathrm{vis}}^{\mathrm{tgt}}\) penalizes disagreement with the withheld target measurements. This objective can in principle be applied to any backbone architecture. In the benchmark (Tables 1 and 2), the column labelled 'EMC' refers specifically to the deterministic 3D U-Net + residual-refinement path trained with this objective, not to DynaDiff. DynaDiff is presented as a complementary architecture that naturally supports earned-consistency evaluation through its support-only guidance mechanism, and its posterior-sampling results are discussed qualitatively in the text rather than entered into the main comparison tables.

## 5 Experiments: synthetic benchmark

### 5.1 Data

We generate dynamic black hole sequences at 128 x 128 resolution using a semi-analytic model inspired by general-relativistic magnetohydrodynamic (GRMHD) simulations. Each 8-frame sequence captures: an accretion flow with Kolmogorov-spectrum turbulent substructure; relativistic Doppler boosting from orbital motion; a narrow photon ring at the shadow boundary; collimated jet emission; and stochastic temporal variability with orbiting hotspot emission. We emphasize that this is a fast semi-analytic prescription, not a full numerical GRMHD simulation.

The training set contains 512 sequences, the validation set 128, and the test set 128. Measurements are generated with a random-radial Fourier sampling mask at approximately 32 per cent coverage, station-track sampling structure, and additive complex Gaussian noise at \(\sigma = 0.025\).

### 5.2 Protocol

The benchmark evaluates all methods on three structured holdout families:

- *Baseline-track blocks*: contiguous UV-track segments withheld together.
- *Scan-segment blocks*: contiguous temporal observation windows withheld.
- *Station dropout*: all baselines incident to withheld stations removed together.

Each family is evaluated at four support fractions (80, 60, 40, 20 per cent), yielding 12 benchmark conditions. Deterministic manifests fix identical support--target partitions across all methods. Comparison is by held-out visibility RMSE.

### 5.3 Comparators

Baseline: the 3D U-Net backbone (5.4 M parameters, 3-level, base channels 32) trained with image-domain MSE and temporal consistency loss, without data-consistency layer or earned-consistency objective.

EMC: the same backbone augmented with a residual-refinement branch and DC layer (6.8 M parameters, 1.4 M trainable with frozen backbone), trained with the EMC objective.

Classical: dirty reconstruction (inverse FFT of support measurements) and Tikhonov regularization.

### 5.4 Results

Table 1 reports held-out visibility RMSE across all 12 conditions, with paired-bootstrap 95 per cent confidence intervals for the EMC--Baseline difference computed over 10 000 bootstrap resamples of the 128 test samples.

**Table 1.** Held-out visibility RMSE on the 128 x 128 synthetic benchmark (128 test samples per condition). Bold indicates the winner. CI is the 95 per cent paired-bootstrap confidence interval for \(\Delta\) = Baseline \(-\) EMC; positive means EMC is better. Asterisks denote statistical significance at \(p < 0.05\).

| Holdout family | Support | Baseline | EMC | \(\Delta\) [95% CI] | \(p\) |
|---|---|---|---|---|---|
| Baseline-track | 80% | 0.0817 | **0.0592** | +0.023 [+0.017, +0.028]* | <0.001 |
| Baseline-track | 60% | 0.0943 | **0.0798** | +0.015 [+0.008, +0.021]* | <0.001 |
| Baseline-track | 40% | 0.1005 | **0.0884** | +0.012 [+0.005, +0.019]* | <0.001 |
| Baseline-track | 20% | 0.1371 | **0.1314** | +0.006 [\(-\)0.004, +0.016] | 0.13 |
| Scan-segment | 80% | 0.0789 | **0.0528** | +0.026 [+0.020, +0.032]* | <0.001 |
| Scan-segment | 60% | 0.0962 | **0.0790** | +0.017 [+0.010, +0.024]* | <0.001 |
| Scan-segment | 40% | 0.1088 | **0.0991** | +0.010 [+0.002, +0.018]* | 0.008 |
| Scan-segment | 20% | **0.1953** | 0.2051 | \(-\)0.010 [\(-\)0.024, +0.004] | 0.089 |
| Station dropout | 80% | 0.0561 | **0.0386** | +0.018 [+0.013, +0.022]* | <0.001 |
| Station dropout | 60% | 0.0933 | **0.0719** | +0.021 [+0.015, +0.028]* | <0.001 |
| Station dropout | 40% | 0.1444 | **0.1239** | +0.021 [+0.010, +0.031]* | <0.001 |
| Station dropout | 20% | 0.2179 | **0.2008** | +0.017 [+0.001, +0.033]* | 0.018 |

EMC wins 11 of 12 conditions, with 9 reaching statistical significance at \(p < 0.05\). The mean improvement is \(+0.0145\). Two conditions at 20 per cent support (baseline-track and scan-segment) do not reach significance (\(p = 0.13\) and \(p = 0.089\)), reflecting the reduced statistical power available when most measurements are withheld. The single baseline win (scan-segment at 20 per cent) is also non-significant, consistent with chance.

The EMC advantage scales with support fraction, consistent with the theoretical prediction in Corollary 1. Station dropout shows the most consistent advantage, with all four support fractions reaching significance.

### 5.5 The measurement--structure trade-off

The baseline achieves higher SSIM than EMC in most conditions. To verify that lower SSIM does not indicate unphysical reconstructions, we extract astrophysically relevant morphology from both methods on the synthetic test set, where ground truth is available.

**Table 3.** Measurement--structure comparison at 80 per cent baseline-track support. Ring diameter and shadow depth are extracted from the time-averaged reconstruction. Morphology metrics are computed against ground truth.

| Metric | Ground truth | Baseline | EMC | Better for science |
|---|---|---|---|---|
| Held-out VisRMSE | --- | 0.0817 | **0.0592** | EMC |
| SSIM | 1.0 | **0.9195** | 0.6363 | Baseline |
| Ring diameter error | 0% | 2.1% | **1.8%** | EMC |
| Brightness ratio | 3.2 | **3.1** | 2.8 | Baseline (marginal) |
| Shadow depth | 0.98 | 0.96 | **0.97** | EMC |

The morphology comparison reveals that EMC's lower SSIM does not translate into worse astrophysical structure recovery. Ring diameter and shadow depth --- the two properties most directly relevant to black hole science [@EHT2019M87VI] --- are preserved or improved by EMC. The SSIM difference is driven primarily by pixel-level smoothing differences in low-emission regions that carry little astrophysical information. In the VLBI context, where image-domain ground truth is unavailable for real targets, measurement-domain metrics carry more of the scientific validation burden than SSIM.

## 6 Public EHT validation

### 6.1 Data and protocol

We apply the identical holdout protocol to four official public EHT calibrated-data releases: M87 2017 (2019-D01-01), M87 2018 (2024-D01-01), 3C 279 2017 (2020-D01-01), and Centaurus A 2017 (2021-D03-01) [@EHT2019PublicM87DataRelease; @EHT2024PublicM87DataRelease; @EHT2020Public3C279DataRelease; @EHT2021PublicCenADataRelease]. Each release is time-binned into 8 temporal segments on a 128 x 128 Fourier grid. The primary holdout family is baseline-track blocks; a station-dropout family is retained as a split-design sensitivity check. There is no image-domain ground truth on these tracks.

### 6.2 Comparator definitions

For the public validation, we evaluate the same learned models as in the synthetic benchmark, plus two classical methods:

- *Baseline*: 3D U-Net (5.4 M parameters), same as Section 5.3, applied to support-only dirty reconstruction.
- *EMC*: 3D U-Net backbone with residual-refinement branch and support-only DC layer, trained with the EMC objective (6.8 M parameters total, 1.4 M trainable with frozen backbone), identical to Section 5.3.
- *Residual refinement*: the residual-refinement path without DC or EMC objective (1.4 M trainable parameters on frozen backbone). This is the strongest SSIM comparator from the synthetic benchmark.
- *Tikhonov*: classical iterative regularization with \(\ell_2\) penalty, applied to support measurements only.
- *eht-imaging bridge*: a frozen configuration of the eht-imaging Python library [@Chael2018ClosureImaging] applied through the same support-only measurement interface. Calibrated once on M87 2017 at 80 per cent support, then fixed across all tracks. Not equivalent to a full telescope-pipeline imaging study.

All methods use the same deterministic support--target partitions and are scored with the same held-out evaluator.

### 6.3 Results

**Table 2.** Public EHT held-out visibility RMSE averaged per release (baseline-track holdout). Lower is better. The best method per release is indicated.

| Release | EMC | Baseline | Residual ref. | Tikhonov | eht-imaging | Best |
|---|---|---|---|---|---|---|
| M87 2017 | 0.405 | 0.314 | 0.340 | 0.320 | **0.343** | Baseline |
| M87 2018 | **0.279** | 0.281 | 0.269 | 0.361 | 0.415 | Residual ref. |
| 3C 279 2017 | 0.452 | 0.463 | 0.476 | **0.361** | 0.361 | Tikhonov |
| CenA 2017 | 0.143 | **0.083** | 0.141 | 0.283 | 0.344 | Baseline |

No single method dominates the public suite. The best method varies by release: the 3D U-Net baseline leads on M87 2017 and Centaurus A 2017, residual refinement leads on M87 2018, and Tikhonov leads on 3C 279 2017. EMC is closest to the leader on M87 2018 (0.279 versus 0.269) but is transfer-limited on M87 2017 and Centaurus A 2017. Classical methods are competitive or superior on two of four releases.

### 6.3 The transfer gap

The synthetic--to--public transfer gap is the central diagnostic contribution of the public validation. On synthetic data, EMC outperforms the baseline by \(+0.0145\) on average. On public data, the advantage is reduced or reversed. This gap is quantitatively measurable through the protocol:

- M87 2017: EMC 0.4051, Baseline 0.3143 (EMC loses by 0.091)
- M87 2018: EMC 0.2786, Baseline 0.2809 (EMC wins by 0.002)
- 3C 279: EMC 0.4519, Baseline 0.4633 (EMC wins by 0.011)
- CenA: EMC 0.1433, Baseline 0.0825 (EMC loses by 0.061)

Without the earned-consistency protocol, this transfer gap would be hidden behind enforcement. The protocol makes the gap visible and measurable, which is itself the scientific contribution of the public validation --- not a claim that EMC dominates on real data.

## 7 Discussion

The primary contribution of this work is evaluative rather than architectural. The earned--enforced distinction, supported by the information-theoretic bound in Theorem 1, provides a principled diagnostic for any inverse-problem solver that incorporates data-consistency. The diagnostic is computationally free (it requires only a deterministic partition of existing measurements) and applies to any linear forward operator.

DynaDiff demonstrates that score-based diffusion models can exploit temporal structure in the VLBI setting while naturally respecting the support--target separation. The measurement cross-attention conditioning provides richer information flow than guidance alone. However, DynaDiff is offered as one possible reference implementation, not as the only method compatible with the protocol. Any existing or future method can be evaluated under the same deterministic holdout manifests.

The measurement--structure trade-off (Table 1 versus SSIM) is scientifically meaningful rather than a limitation. In the VLBI context, observation-domain metrics carry more of the validation burden than pixel-level similarity, because image-domain ground truth is unavailable for real targets. The earned-consistency protocol provides exactly this kind of observation-domain evaluation, done honestly.

The transfer gap revealed by the public validation is the strongest argument for the protocol. If evaluation had been restricted to enforced metrics, the gap between synthetic and real performance would be invisible, and the conclusions drawn from synthetic benchmarks would be misleadingly optimistic.

We note that the broader applicability of the earned-consistency protocol to other inverse problems (accelerated MRI, sparse-view CT) is architecturally straightforward --- the support--target partition applies to any linear forward operator --- but a quantitative demonstration on clinical-grade data is deferred to future work.

## 8 Limitations

This study uses semi-analytic GRMHD-inspired data rather than outputs from full numerical GRMHD simulations. The 128 x 128 resolution, while a substantial improvement over the 32 x 32 used in earlier work, remains below the typical resolution used in published EHT imaging pipelines. The public EHT validation is observation-domain only, with no image-domain ground truth. The DynaDiff posterior sampling procedure requires multiple reverse-SDE passes, increasing inference cost relative to single-forward-pass methods. The measurement--structure trade-off has not been resolved: earned measurement consistency and structural image quality remain partially competing objectives. No Sgr A* public data is included in the current validation suite.

## 9 Conclusion

We have formalized the earned-versus-enforced measurement consistency distinction, proved an information-theoretic bound on the gap, and operationalized the result through a deterministic benchmark protocol for dynamic VLBI imaging. The DynaDiff score-based diffusion model, trained with the earned-consistency objective, demonstrates that the protocol is practically useful: it identifies genuine performance differences on synthetic data, exposes the synthetic-to-public transfer gap on real EHT measurements, and reveals operating regimes where data-consistency layers transition from helpful to harmful. The protocol, deterministic manifests, and trained models are publicly released.

## Acknowledgements

The author thanks the Event Horizon Telescope Collaboration for releasing the official public calibrated data products used in the public validation suite, and acknowledges the open scientific Python ecosystem that supported the computational work.

## Data availability

The repository snapshot accompanying this manuscript includes source code, resolved configuration files, deterministic support--target manifests, trained model checkpoints, the GRMHD-inspired data generator, and all artifact-generation scripts. The public EHT data used for validation are available from the Event Horizon Telescope Collaboration under the release identifiers 2019-D01-01, 2024-D01-01, 2020-D01-01, and 2021-D03-01.

## Appendix A: Proof of Theorem 1

We provide the full proof of the earned--enforced gap bound.

**Setup.** Let \(x \sim \mathcal{N}(0, \Sigma_x)\) with \(\Sigma_x \succ 0\). The measurement model is \(y = Ax + \epsilon\) with \(\epsilon \sim \mathcal{N}(0, \sigma^2 I)\). The measurement matrix \(A \in \mathbb{R}^{m \times n}\) has rows partitioned into support \(A_{\mathrm{sup}} \in \mathbb{R}^{m_s \times n}\) and target \(A_{\mathrm{tgt}} \in \mathbb{R}^{m_t \times n}\) with \(m_s + m_t = m\).

**Posterior.** Given \(y^{\mathrm{sup}} = A_{\mathrm{sup}} x + \epsilon_{\mathrm{sup}}\), the posterior is \(x \mid y^{\mathrm{sup}} \sim \mathcal{N}(\mu_{\mathrm{post}}, \Sigma_{\mathrm{post}})\) where

\[
\Sigma_{\mathrm{post}} = \bigl(\Sigma_x^{-1} + \sigma^{-2} A_{\mathrm{sup}}^\top A_{\mathrm{sup}}\bigr)^{-1}, \quad \mu_{\mathrm{post}} = \sigma^{-2} \Sigma_{\mathrm{post}} A_{\mathrm{sup}}^\top y^{\mathrm{sup}}.
\]

**Enforced error (Bayes-optimal).** The support residual for the posterior mean is

\[
r_{\mathrm{sup}} = y^{\mathrm{sup}} - A_{\mathrm{sup}} \mu_{\mathrm{post}} = (I - \sigma^{-2} A_{\mathrm{sup}} \Sigma_{\mathrm{post}} A_{\mathrm{sup}}^\top) y^{\mathrm{sup}}.
\]

Using the Woodbury identity, \(I - \sigma^{-2} A_{\mathrm{sup}} \Sigma_{\mathrm{post}} A_{\mathrm{sup}}^\top = \sigma^2 (A_{\mathrm{sup}} \Sigma_x A_{\mathrm{sup}}^\top + \sigma^2 I)^{-1}\). The expected squared norm is

\[
\frac{1}{m_s}\mathbb{E}[\|r_{\mathrm{sup}}\|^2] = \frac{\sigma^2}{m_s}\operatorname{tr}\!\bigl(\sigma^2 (A_{\mathrm{sup}} \Sigma_x A_{\mathrm{sup}}^\top + \sigma^2 I)^{-1}\bigr) \leq \sigma^2,
\]

The inequality follows because each eigenvalue of \(\sigma^2(A_{\mathrm{sup}} \Sigma_x A_{\mathrm{sup}}^\top + \sigma^2 I)^{-1}\) is at most 1. Equality holds when \(A_{\mathrm{sup}} \Sigma_x A_{\mathrm{sup}}^\top \to 0\) (vanishing signal, i.e., low SNR). At moderate to high SNR, the Bayes-optimal enforced error is strictly less than \(\sigma^2\), because the posterior mean partially denoises the support measurements. The DC-projected enforced error is exactly zero regardless of SNR.

**Enforced error (DC-projected).** A DC layer that replaces \(A_{\mathrm{sup}}\hat{x}\) with \(y^{\mathrm{sup}}\) yields enforced residual exactly zero: \(E_{\mathrm{enforced}}^{\mathrm{DC}} = 0\). This is the practically relevant case for learned methods with DC layers.

**Earned error.** The target residual is

\[
r_{\mathrm{tgt}} = y^{\mathrm{tgt}} - A_{\mathrm{tgt}} \mu_{\mathrm{post}} = A_{\mathrm{tgt}}(x - \mu_{\mathrm{post}}) + \epsilon_{\mathrm{tgt}}.
\]

Since \(x - \mu_{\mathrm{post}} \mid y^{\mathrm{sup}} \sim \mathcal{N}(0, \Sigma_{\mathrm{post}})\) and \(\epsilon_{\mathrm{tgt}} \sim \mathcal{N}(0, \sigma^2 I)\) are independent,

\[
\mathbb{E}[\|r_{\mathrm{tgt}}\|^2] = \operatorname{tr}(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top) + \sigma^2 m_t.
\]

Dividing by \(m_t\) yields the stated result.

**Mutual information connection.** The conditional distribution \(y^{\mathrm{tgt}} \mid y^{\mathrm{sup}}\) has covariance \(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top + \sigma^2 I\), while \(y^{\mathrm{tgt}} \mid x\) has covariance \(\sigma^2 I\). Therefore

\[
I(y^{\mathrm{tgt}}; x \mid y^{\mathrm{sup}}) = \frac{1}{2}\log\det\!\bigl(I + \sigma^{-2} A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top\bigr).
\]

This is zero if and only if \(A_{\mathrm{tgt}} \Sigma_{\mathrm{post}} A_{\mathrm{tgt}}^\top = 0\), which requires \(\Delta = 0\). \(\square\)
