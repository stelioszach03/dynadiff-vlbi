"""Teacher signal for the :class:`HeavyHitterOracle`.

Given a partial-DFT (or any linear-Gaussian) measurement operator ``A``,
a support set ``S``, and measurement-noise/signal-prior variances, the
teacher returns the ground-truth importance

    I*(j | S) = a_j^H * Sigma_post(S) * a_j

for every candidate row ``j``, where ``Sigma_post(S)`` is the posterior
covariance of the signal given ``y_S``. This is the signal-independent
quantity that appears in the earned-versus-enforced gap bounds
(Theorem 1' and Theorem 2, see ``theory/partial_dft_bound.tex`` and
``theory/oracle_bound.tex``).

``I*`` is purely a function of the UV coverage and the variance ratio
``signal_var / noise_var``; it does **not** depend on the particular
signal sample. The oracle's job is to learn to predict it from the
observed support visibilities, which in practice lets the model
generalise across support fractions and UV geometries.
"""

from __future__ import annotations

import torch


def compute_posterior_covariance(
    A_support: torch.Tensor,
    signal_var: float,
    noise_var: float,
) -> torch.Tensor:
    """Return the Gaussian-prior posterior covariance of the signal.

    ``Sigma_post = (sigma_x^-2 I + sigma_n^-2 * Re(A_S^H A_S))^-1``

    Args:
        A_support: Complex tensor ``[..., M_sup, N]`` of support rows.
        signal_var: Prior variance ``sigma_x^2``.
        noise_var: Measurement-noise variance ``sigma_n^2``.

    Returns:
        Real-valued ``[..., N, N]`` posterior covariance. The returned
        tensor has ``dtype=torch.float32`` or ``float64`` depending on
        the input.
    """
    if signal_var <= 0:
        raise ValueError(f"signal_var must be positive, got {signal_var}")
    if noise_var <= 0:
        raise ValueError(f"noise_var must be positive, got {noise_var}")
    if not torch.is_complex(A_support):
        A_support = A_support.to(torch.complex64)

    N = A_support.shape[-1]
    gram = (A_support.conj().transpose(-2, -1) @ A_support).real
    real_dtype = gram.dtype
    eye = torch.eye(N, dtype=real_dtype, device=gram.device)
    precision = gram / noise_var + eye / signal_var
    return torch.linalg.inv(precision)


def compute_importance_teacher(
    A_query: torch.Tensor,
    A_support: torch.Tensor,
    signal_var: float,
    noise_var: float,
) -> torch.Tensor:
    """Ground-truth importance I*(j | S) for each row j of ``A_query``.

    Uses the Woodbury identity so the cost scales with ``m_sup``, not
    with the signal dimension ``n``. This matters for VLBI-scale problems
    where n can be ~16 384 (128x128 images) but m_sup is at most a few
    thousand.

    Derivation. The signal ``x`` is real-valued with Gaussian prior
    ``N(0, sigma_x^2 I)``; measurement noise is complex Gaussian. The
    posterior precision is
        P_post = sigma_x^-2 * I + sigma_n^-2 * Re(A_S^H A_S)
               = sigma_x^-2 * I + sigma_n^-2 * A_stack^T A_stack
    where ``A_stack = [Re(A_S); Im(A_S)]`` is a real ``[2 m_sup, n]``
    matrix. Applying Woodbury with the thin factor A_stack,
        Sigma_post = sigma_x^2 I - sigma_x^4 A_stack^T M^-1 A_stack,
        M          = sigma_n^2 I + sigma_x^2 A_stack A_stack^T.
    Therefore, for every query row ``a_j``,
        I*(j) = a_j^H Sigma_post a_j
              = sigma_x^2 ||a_j||^2 - sigma_x^4 z_j^H M^-1 z_j,
    where ``z_j = A_stack a_j`` is a ``[2 m_sup]`` complex vector. Since
    M is real symmetric, ``z_j^H M^-1 z_j`` reduces to
    ``Re(z_j)^T M^-1 Re(z_j) + Im(z_j)^T M^-1 Im(z_j)``.

    Args:
        A_query: Complex tensor ``[..., M_q, N]`` of candidate rows.
        A_support: Complex tensor ``[..., M_sup, N]`` of support rows.
        signal_var: Prior variance ``sigma_x^2``.
        noise_var: Measurement-noise variance ``sigma_n^2``.

    Returns:
        Real tensor ``[..., M_q]`` of importance scores (non-negative).
    """
    if signal_var <= 0 or noise_var <= 0:
        raise ValueError("signal_var and noise_var must be positive")
    if not torch.is_complex(A_query):
        A_query = A_query.to(torch.complex64)
    if not torch.is_complex(A_support):
        A_support = A_support.to(torch.complex64)

    # Build the real [2 m_sup, n] stacked support operator (real32/64 only;
    # no complex allocation the size of A_support).
    A_stack = torch.cat([A_support.real, A_support.imag], dim=-2)  # [..., 2 m_sup, n]
    real_dtype = A_stack.dtype
    two_m_sup = A_stack.shape[-2]

    # Compute Z = A_stack @ A_query^T by splitting A_query into (real, imag)
    # and doing two real matmuls. This avoids allocating a complex copy of
    # A_stack, halving the peak memory at EHT scale.
    Z_re = A_stack @ A_query.real.to(real_dtype).transpose(-2, -1)
    Z_im = A_stack @ A_query.imag.to(real_dtype).transpose(-2, -1)

    # M = sigma_n^2 * I + sigma_x^2 * A_stack @ A_stack^T  ([..., 2 m_sup, 2 m_sup] real PD).
    gram = A_stack @ A_stack.transpose(-2, -1)
    eye = torch.eye(two_m_sup, dtype=real_dtype, device=gram.device)
    M = noise_var * eye.expand(gram.shape) + signal_var * gram
    # Free A_stack early to lower the peak.
    del A_stack

    # Solve M @ W = Z in one LU / Cholesky on the stacked [Z_re | Z_im] rhs.
    m_q = Z_re.shape[-1]
    rhs = torch.cat([Z_re, Z_im], dim=-1)  # [..., 2 m_sup, 2 m_q]
    W = torch.linalg.solve(M, rhs)          # [..., 2 m_sup, 2 m_q]
    del rhs, M, gram
    W_re = W[..., :m_q]
    W_im = W[..., m_q:]

    # Column-wise dot product: quad[j] = Z_re[:, j]^T W_re[:, j] + Z_im[:, j]^T W_im[:, j]
    quad = (Z_re * W_re).sum(dim=-2) + (Z_im * W_im).sum(dim=-2)

    # Row-wise squared norm of a_j over the signal dim: ||a_j||^2.
    norm_sq = A_query.abs().pow(2).sum(dim=-1).to(real_dtype)

    return signal_var * norm_sq - (signal_var ** 2) * quad


def compute_importance_teacher_batched(
    A: torch.Tensor,
    support_indices: torch.Tensor,
    target_indices: torch.Tensor,
    signal_var: float,
    noise_var: float,
    batch_chunk_size: int | None = None,
) -> torch.Tensor:
    """Batched teacher: one sample per batch dim.

    Args:
        A: Complex tensor ``[B, M, N]`` of measurement operators.
        support_indices: Long tensor ``[B, M_sup]`` of support row indices.
        target_indices: Long tensor ``[B, M_tgt]`` of target row indices.
        signal_var: Prior variance.
        noise_var: Noise variance.
        batch_chunk_size: Process at most this many batch items at a time,
            recomposing the output at the end. Bounds peak memory at
            ``O(chunk_size * (2 m_sup)^2 + chunk_size * n * m_sup)`` which
            is what matters at EHT scale (n~16 384, m_sup~2 000). If
            ``None``, the full batch is processed in a single pass.

    Returns:
        Real tensor ``[B, M_tgt]`` of teacher importance.
    """
    B, M, N = A.shape
    if support_indices.shape[0] != B or target_indices.shape[0] != B:
        raise ValueError("batch size mismatch across A / support / target")

    batch_range = torch.arange(B, device=A.device).unsqueeze(1)

    def _run(slice_: slice) -> torch.Tensor:
        br = batch_range[slice_]
        A_sup = A[slice_][torch.arange(br.shape[0], device=A.device).unsqueeze(1), support_indices[slice_]]
        A_tgt = A[slice_][torch.arange(br.shape[0], device=A.device).unsqueeze(1), target_indices[slice_]]
        return compute_importance_teacher(A_tgt, A_sup, signal_var, noise_var)

    if batch_chunk_size is None or batch_chunk_size >= B:
        A_sup = A[batch_range, support_indices]
        A_tgt = A[batch_range, target_indices]
        return compute_importance_teacher(A_tgt, A_sup, signal_var, noise_var)

    chunks: list[torch.Tensor] = []
    for start in range(0, B, batch_chunk_size):
        end = min(start + batch_chunk_size, B)
        chunks.append(_run(slice(start, end)))
    return torch.cat(chunks, dim=0)
