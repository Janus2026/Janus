import torch
import triton
import torch_npu
import triton.language as tl
import torch.nn.functional as F
from typing import Optional, Tuple
import pytest

MAX_CORES = 65535


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X_base
    N,  # number of columns in X_base
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    N_CORES: tl.constexpr,
):
    # Map the program id to the row of X_base and Y_base it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    n_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    if row < remain:
        n_iters = n_iters + 1

    for i in tl.range(n_iters):
        X_base = X + (i * BLOCK_ROWS *
                      stride_x_row) + row * stride_x_row + group * N
        Y_base = Y + (i * BLOCK_ROWS *
                      stride_y_row) + row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + (i * BLOCK_ROWS *
                          stride_z_row) + row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + (i * BLOCK_ROWS) + group * M
        Rstd_base = Rstd + (i * BLOCK_ROWS) + group * M
        W_base = W + group * N
        if HAS_BIAS:
            B_base = B + group * N
        # Compute mean and variance
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(X_base + cols, mask=cols < N, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base + row, rstd)
        # Normalize and apply linear transformation
        mask = cols < N
        w = tl.load(W_base + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B_base + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(Y_base + cols, y, mask=mask)


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M if M < MAX_CORES else MAX_CORES, ngroups)

    # Print debug logs
    print(f"grid:{grid}")
     
    print(f"x.shape:{x.shape}")
    print(f"out.shape:{out.shape}")
    print(f"weight.shape:{weight.shape}")
    if bias is not None:
        print(f"bias.shape:{bias.shape}")
    if z is not None:
        print(f"z.shape:{z.shape}")
    if mean is not None:
        print(f"mean.shape:{mean.shape}")
    print(f"rstd.shape:{rstd.shape}")

    print(f"x.stride(0):{x.stride(0)}")
    print(f"out.stride(0):{out.stride(0)}")
    if z is not None:
        print(f"z.stride(0):{z.stride(0)}")

    print(f"M:{M}")
    print(f"group_size:{group_size}")
    print(f"eps:{eps}")

    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            N_CORES=MAX_CORES,
            num_warps=num_warps,
        )
    return out, mean, rstd


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        return y.reshape(x_shape_og)
    
def custom_layer_norm(x, weight, bias=None, eps=1e-6, z=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)

def layer_norm_golden_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    z: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False
) -> torch.Tensor:
    # 1. Input preprocessing
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1]).cpu().contiguous()  # (M, N) = (batch*seq_len, feat_dim)
    M, N = x.shape
    
    # 2. Auxiliary variable initialization
    weight = weight.cpu().contiguous()
    bias = bias.cpu().contiguous() if bias is not None else None
    z = z.reshape(-1, z.shape[-1]).cpu().contiguous() if z is not None else None
    if z is not None:
        assert z.shape == (M, N), f"Z shape {z.shape} must match X shape ({M}, {N})"
    
    # 3. Group configuration
    if group_size is None:
        group_size = N
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    ngroups = N // group_size
    
    # 4. Gating logic (if Z exists and gate is applied before normalization: apply z * sigmoid(z) first)
    if z is not None and not norm_before_gate:
        gate = z * torch.sigmoid(z)  # (M, N)
        x = x * gate
    
    # 5. Grouped normalization
    # Reshape: (M, N) → (M, ngroups, group_size) → (M*ngroups, group_size) (compatible with torch.layer_norm)
    x_grouped = x.unfold(dimension=1, size=group_size, step=group_size)  # (M, ngroups, group_size)
    x_grouped = x_grouped.reshape(-1, group_size)  # (M*ngroups, group_size)
    
    # 6. Core normalization (LayerNorm vs RMSNorm)
    if not is_rms_norm:
        # Standard LayerNorm: mean centering + variance normalization
        x_norm = torch.layer_norm(
            x_grouped, 
            normalized_shape=(group_size,), 
            weight=None,  # Apply weights and biases uniformly later to keep consistency with custom kernel
            bias=None,
            eps=eps
        )
    else:
        # RMSNorm: no centering, only root mean square normalization (x / sqrt(mean(x²) + eps))
        mean_sq = torch.mean(x_grouped ** 2, dim=-1, keepdim=True)  # (M*ngroups, 1)
        x_norm = x_grouped * torch.rsqrt(mean_sq + eps)  # rsqrt = 1/sqrt
    
    # 7. Restore grouped shape: (M*ngroups, group_size) → (M, ngroups, group_size) → (M, N)
    x_norm = x_norm.reshape(M, ngroups, group_size)  # Restore group dimension
    x_norm = x_norm.contiguous().view(M, N)  # Merge groups → (M, N)
    
    # 8. Linear transformation (weights + biases, consistent with custom kernel logic)
    y = x_norm * weight  # (M, N) * (N,) → broadcast multiplication
    if bias is not None:
        y = y + bias
    
    # 9. Gating logic (if Z exists and gate is applied after normalization: apply z * sigmoid(z) after normalization)
    if z is not None and norm_before_gate:
        gate = z * torch.sigmoid(z)  # (M, N)
        y = y * gate
    
    # 10. Restore original shape and return
    return y.reshape(x_shape_og)

def test_custom_layer_norm():
    """
    Test cases: cover key scenarios to verify numerical correctness
    Scenarios: different dimensions, with/without bias, with/without Z branch, 
               gate order, LayerNorm/RMSNorm, grouped normalization
    """
    # Test configurations (cover all key parameter combinations)
    test_cases = [
        # (batch_size, seq_len, feat_dim, has_bias, has_z, norm_before_gate, is_rms_norm, group_size)
        # (2, 8, 128, False, False, True, False, None),    # Basic LayerNorm (no bias, no Z)
        # (2, 8, 128, True, False, True, False, None),     
        # (2, 8, 128, True, True, True, False, None),      
        # (2, 8, 128, True, True, False, False, None),    
        (2, 8, 128, False, True, True, True, None),     
        # (2, 8, 128, True, True, True, False, 128),     
        # (4, 16, 256, True, False, True, False, 64),     
        # (1, 4, 64, False, True, False, True, 32),    
    ]
    
    atol = 1e-3  
    rtol = 1e-3 
    
    # Check if NPU device is available
    has_npu = torch.npu.is_available()
    device = torch.device("npu" if torch.npu.is_available() else "cpu")

    for idx, (batch_size, seq_len, feat_dim, has_bias, has_z, norm_before_gate, is_rms_norm, group_size) in enumerate(test_cases):
        print(f"\nTest case {idx+1}:")
        print(f"Configuration: batch={batch_size}, seq={seq_len}, feat={feat_dim}, bias={has_bias}, Z={has_z}, "
              f"norm_before_gate={norm_before_gate}, RMS={is_rms_norm}, group_size={group_size}")
        
        # 1. Generate random input (generate on CPU then move to NPU)
        torch.manual_seed(42)  # Fix seed for reproducibility
        x = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32)  # (B, S, D)
        weight = torch.randn(feat_dim, dtype=torch.float32)  # Weights (required)
        bias = torch.randn(feat_dim, dtype=torch.float32) if has_bias else None
        z = torch.randn(batch_size, seq_len, feat_dim, dtype=torch.float32) if has_z else None
        eps = 1e-6
        
        # 2. Compute CPU golden output
        golden_output = layer_norm_golden_cpu(
            x=x, weight=weight, bias=bias, eps=eps, z=z,
            group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm
        )
        print(golden_output)
        # 3. Compute custom kernel output (only if NPU is available)
        if has_npu:
            x_npu = x.npu()
            weight_npu = weight.npu()
            bias_npu = bias.npu() if bias is not None else None
            z_npu = z.npu() if z is not None else None
            
            custom_output = custom_layer_norm(
                x=x_npu, weight=weight_npu, bias=bias_npu, eps=eps, z=z_npu,
                group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm
            ).cpu()  # Move back to CPU for comparison
            
            # 4. Numerical verification
            max_abs_err = torch.max(torch.abs(custom_output - golden_output))
            max_rel_err = torch.max(torch.abs(custom_output - golden_output) / (torch.abs(golden_output) + eps))
            
            print(f"  Max absolute error: {max_abs_err:.6f} (threshold: {atol})")
            print(f"  Max relative error: {max_rel_err:.6f} (threshold: {rtol})")
            
            # Assert verification (error within acceptable range)
            assert max_abs_err < atol, f"Absolute error exceeds threshold: {max_abs_err:.6f} > {atol}"
            assert max_rel_err < rtol, f"Relative error exceeds threshold: {max_rel_err:.6f} > {rtol}"
            print("  ✅ Numerical verification passed!")
            
    print("\n" + "="*50)
    print("All test cases completed!")

if __name__ == "__main__":
    test_custom_layer_norm()
