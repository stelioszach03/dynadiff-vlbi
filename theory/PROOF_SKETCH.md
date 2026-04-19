# Proof Sketch: Enforcement-versus-Earning Gap

This note provides a compact information-theoretic motivation for the earned-versus-enforced distinction used in the benchmark. It is a proof sketch rather than a full general theorem, and it is intended to justify the benchmark question rather than to replace the empirical evaluation.

## Setup

Let \(X\) denote the latent dynamic image sequence and let \(Y = \mathcal{F}(X)\) denote its full complex visibility tensor on a fixed measurement grid. Split the observed grid into disjoint support and target components:

\[
Y_{\mathrm{sup}} = M_{\mathrm{sup}} \odot Y, \qquad
Y_{\mathrm{tgt}} = M_{\mathrm{tgt}} \odot Y,
\]

with \(M_{\mathrm{sup}} \cap M_{\mathrm{tgt}} = \varnothing\) and \(|M_{\mathrm{sup}}|/|M| = \alpha\).

## Enforcement-only path

A support-set data-consistency layer forces a prediction \(\hat{Y}\) to satisfy

\[
M_{\mathrm{sup}} \odot \hat{Y} = Y_{\mathrm{sup}}.
\]

This constraint is exact on the support set, but it does not itself determine \(Y_{\mathrm{tgt}}\). Any information about the held-out target coefficients must therefore come from the image prior or model class rather than from the enforcement operator. In other words, the support-set projection removes support residuals, but it does not inject new information about the withheld coefficients.

## Earned path

A model trained with a target-holdout loss explicitly minimizes risk on \(Y_{\mathrm{tgt}}\):

\[
\mathcal{L}_{\mathrm{earned}} = \mathbb{E}\left[\ell\!\left(f(Y_{\mathrm{sup}}), Y_{\mathrm{tgt}}\right)\right].
\]

Under standard risk-minimization arguments, the trained predictor approaches the Bayes regressor for \(Y_{\mathrm{tgt}}\) given \(Y_{\mathrm{sup}}\). It therefore exploits dependencies between support and target coefficients induced by the source distribution \(p(X)\).

## Gap intuition

The unresolved uncertainty after conditioning on the support set is captured by the conditional entropy or conditional covariance of \(Y_{\mathrm{tgt}}\) given \(Y_{\mathrm{sup}}\). When \(\alpha < 1\), this quantity is strictly positive unless the target coefficients are a deterministic function of the support coefficients. That residual uncertainty gives a natural scale for the best achievable held-out prediction error.

This yields the benchmark-level intuition:

\[
\mathbb{E}\big[\mathrm{RMSE}_{\mathrm{tgt}}(f_{\mathrm{enforced}})\big]
\;\ge\;
\mathbb{E}\big[\mathrm{RMSE}_{\mathrm{tgt}}(f_{\mathrm{earned}})\big]
 + \Delta(\alpha),
\]

where \(\Delta(\alpha)\) decreases as \(\alpha \to 1\) and is controlled by the residual support-conditioned uncertainty in the target coefficients. In the stylized Gaussian-source verification implemented in `theory/consistency_bound.py`, the empirical enforcement-versus-earning gap tracks a conditional-RMSE proxy derived from the conditional covariance \(\Sigma(Y_{\mathrm{tgt}} \mid Y_{\mathrm{sup}})\).

## Interpretation for the paper

The purpose of this sketch is not to claim a universal theorem for every reconstruction architecture. Its role is narrower and more defensible: it motivates why the benchmark asks a meaningful scientific question at all. A data-consistency layer can make observed-support agreement look excellent, yet the held-out target coefficients still probe whether a method has learned to extrapolate beyond the enforced mask.
