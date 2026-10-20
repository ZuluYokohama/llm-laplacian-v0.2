"""AEI node culling, Suzuki recalibration, and JL effective-resistance sparsification."""

from __future__ import annotations

import math

import torch

from laplacian.operators_torch import jl_dimension


def _log(line):
    print(line, flush=True)
    try:
        from laplacian.runtime import RUNS

        path = RUNS / "heartbeat.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line + "\n", encoding="utf-8")
    except OSError:
        return


def fiedler_from_adjacency(adjacency, steps=30):
    """Projected power iteration on a dense nonnegative adjacency."""
    degree = adjacency.sum(dim=1)
    inv = torch.zeros_like(degree)
    positive = degree > 0
    inv[positive] = degree[positive].rsqrt()
    normal = inv[:, None] * adjacency * inv[None, :]
    v1 = degree.clamp_min(0).sqrt()
    norm = torch.linalg.vector_norm(v1)
    if float(norm) < 1e-8:
        return adjacency.new_zeros(adjacency.shape[0]), 0.0, True
    v1 = v1 / norm

    def matvec(vector):
        return vector - normal @ vector

    # Spectral-norm estimate of the normalized Laplacian, then the shift ||L|| I - L.
    probe = torch.linspace(-0.5, 0.5, adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    probe = probe / probe.norm()
    spectral = 0.0
    for _step in range(12):
        image = matvec(probe)
        spectral = float(torch.linalg.vector_norm(image).item())
        if spectral < 1e-12:
            break
        probe = image / spectral
    shift = spectral + 1e-6
    vector = torch.arange(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    vector = vector - torch.dot(vector, v1) * v1
    vector = vector / vector.norm().clamp_min(1e-8)
    for _step in range(steps):
        image = shift * vector - matvec(vector)
        image = image - torch.dot(image, v1) * v1
        norm = torch.linalg.vector_norm(image)
        if float(norm) < 1e-10:
            break
        vector = image / norm
    rayleigh = float(torch.dot(vector, matvec(vector)).item())
    return vector, rayleigh, rayleigh < 1e-8


def adjacency_edge_index(adjacency, fiedler):
    difference = (fiedler[:, None] - fiedler[None, :]).abs()
    return (adjacency * difference).sum(dim=1)


def correlation_adjacency(second, mean, count):
    """Absolute Pearson correlation with a zero diagonal. Zero-variance rows stay zero."""
    second = second / count
    mean = mean / count
    covariance = second - mean[:, None] * mean[None, :]
    variance = covariance.diag().clamp_min(0)
    scale = variance.sqrt()
    inv = torch.zeros_like(scale)
    positive = scale > 1e-8
    inv[positive] = 1.0 / scale[positive]
    corr = covariance * inv[:, None] * inv[None, :]
    corr = corr.abs()
    corr.fill_diagonal_(0)
    corr = torch.where(positive[:, None] & positive[None, :], corr, torch.zeros_like(corr))
    return corr


def select_peripheral(scores, alive, quota):
    """Indices of the lowest-AEI alive neurons. Dead neurons are not selected again."""
    masked = scores.clone()
    masked[~alive] = float("inf")
    available = int(alive.sum().item())
    take = min(quota, max(available - 1, 0))
    if take <= 0:
        return scores.new_zeros(0, dtype=torch.long)
    _values, indices = torch.topk(masked, take, largest=False)
    return indices


def suzuki_recalibrate(down, sigma, survivors, tau):
    """W_out[:, J] <- (W_out @ Sigma[:, J] (Sigma_JJ + tau I)^{-1})."""
    if survivors.numel() == 0:
        return {"tau": tau, "solved": False}
    block = sigma[survivors][:, survivors]
    eye = torch.eye(block.shape[0], device=block.device, dtype=block.dtype)
    ridge = block + tau * eye
    cross = sigma[:, survivors]
    try:
        solved = torch.linalg.solve(ridge, cross.transpose(0, 1)).transpose(0, 1)
    except RuntimeError:
        solved = torch.linalg.solve(ridge + 1e-2 * eye, cross.transpose(0, 1)).transpose(0, 1)
        tau = tau + 1e-2
    updated = down.weight.detach() @ solved
    with torch.no_grad():
        rebuilt = torch.zeros_like(down.weight)
        rebuilt[:, survivors] = updated
        down.weight.copy_(rebuilt)
    info = float(torch.trace(sigma - solved @ cross.transpose(0, 1)).abs().item())
    return {"tau": tau, "solved": True, "info_trace": info, "survivors": int(survivors.numel())}


def _bipartite_matvec(x_src, x_dst, weight):
    """Graph Laplacian matvec for edge weights `weight` shaped [dst, src]."""
    dst = weight @ x_src
    src = weight.transpose(0, 1) @ x_dst
    deg_src = weight.sum(dim=0)
    deg_dst = weight.sum(dim=1)
    return deg_src[:, None] * x_src - src, deg_dst[:, None] * x_dst - dst, deg_src, deg_dst


def _cg_bipartite(weight, rhs_src, rhs_dst, tol=1e-4, maxiter=80):
    """Jacobi CG on the singular bipartite Laplacian. Columns are independent RHS.

    Language-model steps may leave TF32 matmul enabled. This solve is a spectral
    penalty and stays in IEEE fp32 for the duration of the call.
    """
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        deg_src = weight.sum(dim=0).clamp_min(1e-8)
        deg_dst = weight.sum(dim=1).clamp_min(1e-8)
        x_src = torch.zeros_like(rhs_src)
        x_dst = torch.zeros_like(rhs_dst)

        def project(src, dst):
            # One constant kernel across both parts.
            total = src.sum(dim=0, keepdim=True) + dst.sum(dim=0, keepdim=True)
            shift = total / (src.shape[0] + dst.shape[0])
            return src - shift, dst - shift

        def matvec(src, dst):
            out_src, out_dst, _ds, _dd = _bipartite_matvec(src, dst, weight)
            return project(out_src, out_dst)

        r_src, r_dst = project(rhs_src, rhs_dst)
        z_src = r_src / deg_src[:, None]
        z_dst = r_dst / deg_dst[:, None]
        p_src, p_dst = z_src.clone(), z_dst.clone()
        rz = (r_src * z_src).sum(dim=0) + (r_dst * z_dst).sum(dim=0)
        for _it in range(maxiter):
            ap_src, ap_dst = matvec(p_src, p_dst)
            denom = (p_src * ap_src).sum(dim=0) + (p_dst * ap_dst).sum(dim=0)
            # A nonpositive denominator is divergence, not a slow solve. Stop
            # before alpha explodes. maxiter and the JL width stay unchanged.
            if (not torch.isfinite(denom).all()) or float(denom.min()) <= 0.0:
                break
            alpha = rz / denom
            x_src = x_src + p_src * alpha
            x_dst = x_dst + p_dst * alpha
            r_src = r_src - ap_src * alpha
            r_dst = r_dst - ap_dst * alpha
            r_src, r_dst = project(r_src, r_dst)
            residual = torch.maximum(r_src.norm(dim=0), r_dst.norm(dim=0)).max()
            if float(residual) < tol:
                return x_src, x_dst, float(residual)
            z_src = r_src / deg_src[:, None]
            z_dst = r_dst / deg_dst[:, None]
            rz_next = (r_src * z_src).sum(dim=0) + (r_dst * z_dst).sum(dim=0)
            beta = rz_next / rz.clamp_min(1e-12)
            p_src = z_src + p_src * beta
            p_dst = z_dst + p_dst * beta
            rz = rz_next
        residual = torch.maximum(r_src.norm(dim=0), r_dst.norm(dim=0)).max()
        return x_src, x_dst, float(residual)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def resistance_sketch(weight, mask, eps, batch_columns, log_prefix):
    """JL embedding of effective resistance for the bipartite graph of `weight`.

    Culled neurons are isolated. They are removed before the solve, and k uses
    that live order. Epsilon is unchanged. Resistances scatter back to the
    full [dst, src] shape; isolated rows and columns stay zero.
    """
    conduct = weight.detach().abs() * mask
    n_dst, n_src = conduct.shape
    live_dst = conduct.sum(dim=1) > 0
    live_src = conduct.sum(dim=0) > 0
    n_live = int(live_dst.sum().item()) + int(live_src.sum().item())
    if int(live_dst.sum().item()) == 0 or int(live_src.sum().item()) == 0:
        _log(f"PULSE {log_prefix} cg=0/0 residual=0.000e+00 empty")
        return torch.zeros(n_dst, n_src, device=weight.device, dtype=torch.float32), 0
    dst_index = live_dst.nonzero(as_tuple=False).flatten()
    src_index = live_src.nonzero(as_tuple=False).flatten()
    reduced = conduct[dst_index][:, src_index]
    k = jl_dimension(max(n_live, 2), eps)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(n_live * 10007 + k)
    n_dst_live = int(dst_index.numel())
    n_src_live = int(src_index.numel())
    z2_src = torch.zeros(n_src_live, device=weight.device, dtype=torch.float32)
    z2_dst = torch.zeros(n_dst_live, device=weight.device, dtype=torch.float32)
    dots = torch.zeros(n_dst_live, n_src_live, device=weight.device, dtype=torch.float32)
    sqrt_w = reduced.clamp_min(0).sqrt()
    done = 0
    columns = batch_columns
    while done < k:
        width = min(columns, k - done)
        try:
            signs = torch.randint(0, 2, (width, n_dst_live, n_src_live), generator=generator, dtype=torch.float32)
            signs = signs.mul_(2).sub_(1).to(weight.device)
            scale = 1.0 / math.sqrt(k)
            rhs_src = torch.einsum("whi,hi->wi", signs, sqrt_w).transpose(0, 1) * scale
            rhs_dst = -torch.einsum("whi,hi->wh", signs, sqrt_w).transpose(0, 1) * scale
            sol_src, sol_dst, cg_residual = _cg_bipartite(reduced, rhs_src, rhs_dst)
            z2_src = z2_src + sol_src.pow(2).sum(dim=1)
            z2_dst = z2_dst + sol_dst.pow(2).sum(dim=1)
            dots = dots + sol_dst @ sol_src.transpose(0, 1)
            done += width
            _log(f"PULSE {log_prefix} cg={done}/{k} residual={cg_residual:.3e}")
            del signs, rhs_src, rhs_dst, sol_src, sol_dst
        except torch.cuda.OutOfMemoryError:
            if columns <= 1:
                raise
            columns = max(1, columns // 2)
            _log(f"PULSE {log_prefix} cg={done}/{k} batch_columns={columns}")
            if weight.is_cuda:
                torch.cuda.empty_cache()
    reduced_resistance = (z2_dst[:, None] + z2_src[None, :] - 2.0 * dots).clamp_min(0)
    resistance = torch.zeros(n_dst, n_src, device=weight.device, dtype=torch.float32)
    resistance[dst_index[:, None], src_index[None, :]] = reduced_resistance
    return resistance, k


def sparsify_matrix(weight, mask, eps, retain_floor, mode, log_prefix, batch_columns=32):
    """Rescale or drop edges using JL leverage scores. Returns measured keep fraction."""
    device = weight.device
    with torch.no_grad():
        live = (mask > 0).to(weight.dtype)
        nnz = int(live.sum().item())
        if nnz <= 1:
            return {
                "nnz_before": nnz,
                "nnz_after": nnz,
                "kept_fraction": 1.0,
                "k": 0,
                "mode": mode,
                "degenerate_sample": False,
            }
        resistance, k = resistance_sketch(weight, live, eps, batch_columns, log_prefix)
        leverage = (weight.detach().abs() * live * resistance).reshape(-1)
        leverage = torch.where(live.reshape(-1) > 0, leverage, torch.zeros_like(leverage))
        total = leverage.sum().clamp_min(1e-12)
        probability = leverage / total
        n_vertices = weight.shape[0] + weight.shape[1]
        budget = math.ceil(n_vertices * math.log(max(n_vertices, 2)) / (eps * eps))
        floor = math.ceil(retain_floor * nnz)
        retain = min(nnz, max(budget, floor))
        flat_live = live.reshape(-1) > 0
        scores = probability.clone()
        scores[~flat_live] = -1.0
        finite = bool(torch.isfinite(probability).all().item())
        mass = probability.clamp_min(0).sum()
        degenerate_sample = mode != "topk" and ((not finite) or float(mass) <= 0.0)
        if mode == "topk" or degenerate_sample:
            safe_scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1.0))
            chosen = torch.topk(safe_scores, retain).indices
            counts = torch.zeros_like(probability)
            counts[chosen] = 1.0
            if degenerate_sample:
                _log(f"PULSE {log_prefix} degenerate_sample=1")
        else:
            probs = probability.clamp_min(0)
            probs = probs / probs.sum()
            draws = torch.multinomial(probs, retain, replacement=True)
            counts = torch.bincount(draws, minlength=probs.numel()).to(probs.dtype)
            unique = int((counts > 0).sum().item())
            if unique < floor:
                extra = torch.topk(scores, floor).indices
                counts[extra] = torch.maximum(counts[extra], torch.ones_like(counts[extra]))
        flat = weight.detach().reshape(-1)
        if degenerate_sample:
            # A zero or non-finite leverage vector must not enter multinomial,
            # and it must not rescale the kept entries by 1/0.
            signed = torch.where(counts > 0, flat, torch.zeros_like(flat))
        else:
            r = max(int(retain), 1)
            rescaled = counts * (flat.abs() / (r * probability.clamp_min(1e-12)))
            rescaled = torch.where(counts > 0, rescaled, torch.zeros_like(rescaled))
            signed = torch.sign(flat) * rescaled
        new_weight = signed.view_as(weight)
        new_mask = (counts > 0).to(weight.dtype).view_as(mask)
        weight.copy_(new_weight)
        mask.copy_(new_mask)
        kept = int(new_mask.sum().item())
        return {
            "nnz_before": nnz,
            "nnz_after": kept,
            "kept_fraction": kept / max(nnz, 1),
            "k": k,
            "mode": mode,
            "retain_target": retain,
            "degenerate_sample": degenerate_sample,
        }


def apply_neuron_cull(block, remove):
    with torch.no_grad():
        block.mlp.neuron_mask[remove] = 0
        block.mlp.up.weight[remove] = 0
        block.mlp.up_mask[remove] = 0
        block.mlp.down.weight[:, remove] = 0
        block.mlp.down_mask[:, remove] = 0


def layer_pulse(block, second, mean, count, quota, eps, retain_floor, mode, layer_index, tau=1e-3):
    adjacency = correlation_adjacency(second, mean, count)
    fiedler, lam, degenerate = fiedler_from_adjacency(adjacency, steps=30)
    report = {"layer": layer_index, "lambda2": lam, "degenerate": degenerate, "quota": int(quota)}
    if degenerate:
        report["removed"] = 0
        return report
    aei = adjacency_edge_index(adjacency, fiedler)
    alive = block.mlp.neuron_mask > 0
    remove = select_peripheral(aei, alive, quota)
    survivors = torch.where(alive)[0]
    if remove.numel():
        drop = torch.zeros(block.mlp.hidden, dtype=torch.bool, device=remove.device)
        drop[remove] = True
        survivors = survivors[~drop[survivors]]
    sigma = second / count
    report["suzuki"] = suzuki_recalibrate(block.mlp.down, sigma, survivors, tau)
    apply_neuron_cull(block, remove)
    report["removed"] = int(remove.numel())
    report["alive"] = int((block.mlp.neuron_mask > 0).sum().item())
    report["up"] = sparsify_matrix(
        block.mlp.up.weight, block.mlp.up_mask, eps, retain_floor, mode, f"layer={layer_index} matrix=up"
    )
    report["down"] = sparsify_matrix(
        block.mlp.down.weight, block.mlp.down_mask, eps, retain_floor, mode, f"layer={layer_index} matrix=down"
    )
    attn = block.attn
    report["attn"] = [
        sparsify_matrix(attn.qkv.weight[: block.mlp.up.in_features], attn.q_mask, eps, retain_floor, mode, f"layer={layer_index} matrix=q"),
        sparsify_matrix(
            attn.qkv.weight[block.mlp.up.in_features : 2 * block.mlp.up.in_features],
            attn.k_mask,
            eps,
            retain_floor,
            mode,
            f"layer={layer_index} matrix=k",
        ),
        sparsify_matrix(attn.qkv.weight[2 * block.mlp.up.in_features :], attn.v_mask, eps, retain_floor, mode, f"layer={layer_index} matrix=v"),
        sparsify_matrix(attn.proj.weight, attn.o_mask, eps, retain_floor, mode, f"layer={layer_index} matrix=o"),
    ]
    return report


def collect_moments(model, batch_iter, sequences, microbatch):
    """Stream MLP activation second moments. Does not store the activation matrix."""
    device = next(model.parameters()).device
    width = [block.mlp.hidden for block in model.blocks]
    sums = [torch.zeros(n, device=device) for n in width]
    grams = [torch.zeros(n, n, device=device) for n in width]
    seen = 0
    token_count = 0
    model.eval()
    for batch in batch_iter:
        if seen >= sequences:
            break
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch)
        seen += int(batch.shape[0])
        tokens_here = 0
        for index, hidden in enumerate(output["hidden"]):
            flat = hidden.float().reshape(-1, width[index])
            tokens_here = int(flat.shape[0])
            sums[index] = sums[index] + flat.sum(dim=0)
            grams[index] = grams[index] + flat.transpose(0, 1) @ flat
        token_count += tokens_here
        _log(f"PULSE calibrate sequences={seen}/{sequences} tokens={token_count}")
        if seen >= sequences:
            break
    return grams, sums, max(token_count, 1)


def cull_pulse(model, batch_iter, sequences, fraction, eps, retain_floor, mode, microbatch):
    grams, sums, count = collect_moments(model, batch_iter, sequences, microbatch)
    reports = []
    for index, block in enumerate(model.blocks):
        quota = max(1, int(math.floor(block.mlp.hidden * fraction)))
        reports.append(
            layer_pulse(block, grams[index], sums[index], count, quota, eps, retain_floor, mode, index)
        )
        # Drop the tensor but keep the slot. Deleting by index shifts the
        # list and feeds later layers the wrong moments, then raises IndexError.
        grams[index] = None
        sums[index] = None
        torch.cuda.empty_cache()
    model.train()
    return reports


def scrub_optimizer(model, optimizer):
    """Zero Adam moments on entries the masks have already removed."""
    if optimizer is None:
        return

    def _zero(parameter, row_keep=None, col_keep=None):
        state = optimizer.state.get(parameter)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state.get(key)
            if moment is None:
                continue
            if row_keep is not None:
                moment[row_keep <= 0] = 0
            if col_keep is not None:
                moment[:, col_keep <= 0] = 0

    for block in model.blocks:
        alive = block.mlp.neuron_mask
        _zero(block.mlp.up.weight, row_keep=alive)
        _zero(block.mlp.down.weight, col_keep=alive)
        up_state = optimizer.state.get(block.mlp.up.weight)
        down_state = optimizer.state.get(block.mlp.down.weight)
        for state, mask in ((up_state, block.mlp.up_mask), (down_state, block.mlp.down_mask)):
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state.get(key)
                if moment is not None:
                    moment[mask <= 0] = 0
        for parameter, mask in (
            (block.attn.qkv.weight, block.attn._qkv_mask()),
            (block.attn.proj.weight, block.attn.o_mask),
        ):
            state = optimizer.state.get(parameter)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state.get(key)
                if moment is not None and moment.shape == mask.shape:
                    moment[mask <= 0] = 0


def node_only_pulse(model, batch_iter, sequences, fraction):
    """Drop the lowest-AEI neurons. No Suzuki update and no edge sparsification."""
    grams, sums, count = collect_moments(model, batch_iter, sequences, 0)
    reports = []
    for index, block in enumerate(model.blocks):
        quota = max(1, int(math.floor(block.mlp.hidden * fraction)))
        adjacency = correlation_adjacency(grams[index], sums[index], count)
        fiedler, lam, degenerate = fiedler_from_adjacency(adjacency, steps=30)
        report = {"layer": index, "lambda2": lam, "degenerate": bool(degenerate), "quota": int(quota)}
        if degenerate:
            report["removed"] = 0
        else:
            alive = block.mlp.neuron_mask > 0
            remove = select_peripheral(adjacency_edge_index(adjacency, fiedler), alive, quota)
            apply_neuron_cull(block, remove)
            report["removed"] = int(remove.numel())
            report["alive"] = int((block.mlp.neuron_mask > 0).sum().item())
        reports.append(report)
        _log(f"PULSE layer={index} nodes={report['removed']} mode=node")
    model.train()
    return reports
