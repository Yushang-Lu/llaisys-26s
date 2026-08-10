import llaisys
import torch
from test_utils import *
import argparse
import ctypes


def test_basic_runtime_api(device_name: str = "cpu"):

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")
    if ndev == 0:
        print("     Skipped")
        return

    for i in range(ndev):
        print(f"Testing device {i}...")
        api.set_device(i)
        test_memcpy(api, 1024 * 1024)
        test_async_memcpy(api, 1024 * 1024)

        print("     Passed")


def test_memcpy(api, size_bytes: int):
    a = torch.zeros((size_bytes,), dtype=torch.uint8, device=torch_device("cpu"))
    b = torch.ones_like(a)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)
    try:
        # a -> device_a -> device_b -> b
        api.memcpy_sync(
            device_a,
            a.data_ptr(),
            size_bytes,
            llaisys.MemcpyKind.H2D,
        )
        api.memcpy_sync(
            device_b,
            device_a,
            size_bytes,
            llaisys.MemcpyKind.D2D,
        )
        api.memcpy_sync(
            b.data_ptr(),
            device_b,
            size_bytes,
            llaisys.MemcpyKind.D2H,
        )
        torch.testing.assert_close(a, b)
    finally:
        api.free_device(device_b)
        api.free_device(device_a)


def test_async_memcpy(api, size_bytes: int):
    host_src = api.malloc_host(size_bytes)
    host_dst = api.malloc_host(size_bytes)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)
    stream = api.create_stream()
    try:
        ctypes.memset(host_src, 0x5A, size_bytes)
        ctypes.memset(host_dst, 0, size_bytes)
        api.memcpy_async(
            device_a,
            host_src,
            size_bytes,
            llaisys.MemcpyKind.H2D,
            stream,
        )
        api.memcpy_async(
            device_b,
            device_a,
            size_bytes,
            llaisys.MemcpyKind.D2D,
            stream,
        )
        api.memcpy_async(
            host_dst,
            device_b,
            size_bytes,
            llaisys.MemcpyKind.D2H,
            stream,
        )
        api.stream_synchronize(stream)
        assert ctypes.string_at(host_dst, size_bytes) == bytes([0x5A]) * size_bytes
    finally:
        api.destroy_stream(stream)
        api.free_device(device_b)
        api.free_device(device_a)
        api.free_host(host_dst)
        api.free_host(host_src)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)
    
    print("\033[92mTest passed!\033[0m\n")
