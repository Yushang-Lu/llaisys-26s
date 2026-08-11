import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, zero_tensor


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    vals, vals_ = random_tensor(shape, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    assert check_equal(max_val_, max_val, strict=True)
    assert check_equal(max_idx_, max_idx, strict=True)

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name,
        )


def test_argmax_values(values, dtype_name, device_name):
    vals, vals_ = random_tensor(values.shape, dtype_name, device_name)
    vals.copy_(values.to(device=vals.device, dtype=vals.dtype))
    api = llaisys.RuntimeAPI(vals_.device_type())
    api.memcpy_sync(
        vals_.data_ptr(),
        vals.data_ptr(),
        vals.numel() * vals.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)
    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    assert check_equal(max_idx_, max_idx, strict=True)
    if torch.isnan(max_val).any():
        actual_max = torch.empty_like(max_val)
        api.memcpy_sync(
            actual_max.data_ptr(),
            max_val_.data_ptr(),
            actual_max.numel() * actual_max.element_size(),
            llaisys.MemcpyKind.D2D,
        )
        assert torch.isnan(actual_max).all()
    else:
        assert check_equal(max_val_, max_val, strict=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(4,), (4096,)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.argmax on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_argmax(shape, dtype_name, args.device, args.profile)

    if args.device in ("nvidia", "metax"):
        large_values = -torch.arange(1, 151937, dtype=torch.float32)
        large_values[100000] = 3.0

        tied_values = torch.full((8192,), -2.0)
        tied_values[37] = 5.0
        tied_values[7001] = 5.0

        negative_values = -torch.arange(1, 10001, dtype=torch.float32)

        nan_values = torch.full((151936,), -1.0)
        nan_values[17] = 100.0
        nan_values[4097] = float("nan")
        nan_values[120000] = float("nan")

        edge_cases = (
            ("large vocabulary", large_values),
            ("cross-block tie", tied_values),
            ("all negative", negative_values),
            ("NaN precedence", nan_values),
        )
        for case_name, values in edge_cases:
            for dtype_name in testDtype:
                print(
                    f"   {case_name} shape {tuple(values.shape)} dtype <{dtype_name}>"
                )
                test_argmax_values(values, dtype_name, args.device)

    print("\033[92mTest passed!\033[0m\n")
