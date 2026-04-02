import triton
import triton.language as tl
import torch
import pytest

@triton.jit(do_not_specialize=['head_num'])
def rope_inplace_kernel(
    x_ptr, # [bs, qhead, 512]
    sin_ptr,
    cos_ptr,
    head_num, # head_num  q_rope and k_rope use diffrent head_num, so it need to be virant
    x_stride,
    cos_stride,
    hidden_size: tl.constexpr,
    rope_dim: tl.constexpr
):
    cur_b = tl.program_id(0)
    dim_start = hidden_size - rope_dim
    # load x
    offset_x = cur_b * x_stride + dim_start + tl.arange(0, rope_dim)
    x = tl.load(x_ptr + offset_x).to(tl.float32)
    # load sin cos
    offset_sin_cos = cur_b // head_num * cos_stride + tl.arange(0, rope_dim)
    sin = tl.load(sin_ptr + offset_sin_cos).to(tl.float32)
    cos = tl.load(cos_ptr + offset_sin_cos).to(tl.float32)
    
    even = tl.extract_slice(x, [0], [rope_dim // 2], [2])
    odd = tl.extract_slice(x, [1], [rope_dim // 2], [2])
    odd = -odd

    x_rotate = tl.zeros([rope_dim], dtype=tl.float32)
    x_rotate = tl.insert_slice(x_rotate, odd, [0], [rope_dim // 2], [2])
    x_rotate = tl.insert_slice(x_rotate, even, [1], [rope_dim // 2], [2])

    out = x * cos + x_rotate * sin
    tl.store(x_ptr + offset_x, out.to(tl.bfloat16))

def triton_apply_rope_partial_inplace(x, sin, cos):
    rope_dim = sin.shape[-1]
    org_shape = x.shape
    if x.dim() == 2:
        bsz, hidden_size = x.shape
        head_num = 1
    elif x.dim() == 3:
        bsz, head_num, hidden_size = x.shape
        x = x.view(-1, hidden_size)
    else:
        raise NotImplementedError(f"Unsupported dimension: {x.dim()}")
    cores = bsz * head_num
    assert cores < 65535
    rope_inplace_kernel[(cores,)](
        x, 
        sin, 
        cos, 
        head_num,
        x.stride(0), 
        cos.stride(0), 
        hidden_size, 
        rope_dim)
    return x.view(org_shape)

def rope_ref(x, sin, cos, rope_dim):
    total_dim = x.shape[-1]
    orig_dtype = x.dtype
    pre = x[..., :-rope_dim] if rope_dim < total_dim else None
    x_rope = x[..., -rope_dim:]
    if sin.dim() == 2:
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)

    x_even = x_rope[..., ::2]
    x_odd = x_rope[..., 1::2]

    x_rotate = torch.empty_like(x_rope)
    x_rotate[..., ::2] = -x_odd
    x_rotate[..., 1::2] = x_even

    out_rope = x_rope * cos + x_rotate * sin
    out_rope = out_rope.to(orig_dtype)

    if pre is not None:
        out = torch.cat([pre, out_rope], dim=-1)
    else:
        out = out_rope
    return out

@pytest.mark.parametrize("batch, head", [(1, 8), (4, 8), (8, 8), (1, 1), (4, 1), (8, 1)])
def test_rope_inplace(batch, head, hidden_size = 512, rope_dim = 64, itype = torch.bfloat16):
    assert rope_dim % 2 == 0
    assert hidden_size >= rope_dim
    x = torch.randn(batch, head, hidden_size, dtype = itype)
    sin = torch.randn(batch, rope_dim, dtype = itype)
    cos = torch.randn(batch, rope_dim, dtype = itype)
    out_ref = rope_ref(x, sin, cos, rope_dim)
    x_npu = x.npu()
    sin_npu = sin.npu()
    cos_npu = cos.npu()
    out_npu = triton_apply_rope_partial_inplace(x_npu, sin_npu, cos_npu)
    assert torch.allclose(out_ref, out_npu.cpu(), atol=5e-3, rtol=1e-2)
    print(f"test passed for rope_inplace, batch = {batch}, head = {head}, hidden_size = {hidden_size}, rope_dim = {rope_dim}")

if __name__ == "__main__":
    pass
