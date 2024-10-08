import torch
import numpy as np
import random
import yaml
from functools import wraps
import argparse

enable_profile_shape = False


def profile_shape(enable_shape_prof):
    def profile_benchmark(benchmark_func):
        if not enable_shape_prof:
            return benchmark_func

        @wraps(benchmark_func)
        def wrapper(*args, **kwargs):
            with torch.autograd.profiler.profile(
                enabled=True, record_shapes=True
            ) as prof:
                elapsed_time = benchmark_func(*args, **kwargs)
            print(prof.key_averages(group_by_input_shape=True).table())
            return elapsed_time

        return wrapper

    return profile_benchmark


def gen_rand_tensor(tensor):
    if tensor.dtype == torch.bool:
        tensor = torch.randint(0, 2, tensor.size(), dtype=torch.bool, device="cuda")
    elif tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        min_val, max_val = tensor.min(), tensor.max()
        if min_val == max_val:
            return tensor
        else:
            tensor = torch.randint_like(
                tensor, low=min_val, high=max_val, device="cuda"
            )
    else:
        tensor = torch.rand(tensor.size(), device="cuda")
    return tensor


@profile_shape(enable_profile_shape)
def benchmark_function(func, args=(), kwargs={}, n_warmup=3, n_iter=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = np.zeros(n_warmup + n_iter)
    for i in range(n_warmup + n_iter):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = gen_rand_tensor(arg)
        for kwarg in kwargs:
            if isinstance(arg, torch.Tensor):
                kwarg = gen_rand_tensor(kwarg)

        with torch.no_grad():
            start.record()
            func(*args, **kwargs)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[n_warmup:]
    elapsed_time = np.amax(times) / 1000
    return elapsed_time


def benchmark_bitwise_and(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda") > 0.5
    else:
        a = torch.rand(a_size, device="cuda") > 0.5

    if len(b_size) == 0:
        b = torch.rand(1, device="cuda") > 0.5
    else:
        b = torch.rand(b_size, device="cuda") > 0.5

    elapsed_time = benchmark_function(
        torch.ops.aten.bitwise_and, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for bitwise_and (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_bitwise_or(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda") > 0.5
    else:
        a = torch.rand(a_size, device="cuda") > 0.5

    if len(b_size) == 0:
        b = torch.rand(1, device="cuda") > 0.5
    else:
        b = torch.rand(b_size, device="cuda") > 0.5

    elapsed_time = benchmark_function(
        torch.ops.aten.bitwise_or, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for bitwise_or (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_div(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = torch.rand(1, device="cuda")
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.div, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for div (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_div_(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = torch.rand(1, device="cuda")
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.div_, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for div_ (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_eq(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = torch.rand(1, device="cuda")
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.eq, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for eq (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_exponential_(size, n_warmup, n_iter):
    a_size = size[0]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.exponential_, args=(a, 1), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for exponential_ (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_fill_(size, n_warmup, n_iter, dtype=torch.half):
    a_size = size[0]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, dtype=dtype, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.fill_, args=(a, 0), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for fill_ (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_ge(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = 0
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.ge, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for ge (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_index(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        raise ValueError("The indexed tensor cannot be zero dimention.")
    else:
        a = torch.rand(a_size, device="cuda")

    if (len(b_size) != 1) or (b_size[0] > len(a_size)):
        raise ValueError("The index size is illegitimate.")
    else:
        rand_idx = []
        for i in range(b_size[0]):
            rand_idx.append(random.randint(0, a_size[i]))
        b = torch.tensor(rand_idx, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.index, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for index (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_log(size, n_warmup, n_iter):
    a_size = size[0]
    if len(a_size) == 0:
        raise ValueError("The input tensor cannot be zero dimention.")
    else:
        a = torch.rand(a_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.log, args=(a), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for log (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_lt(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        a = torch.rand(1, device="cuda")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = 0
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.lt, args=(a, b), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for lt (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_masked_fill_(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        raise ValueError("The input tensor cannot be zero dimention.")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        raise ValueError("The mask tensor cannot be zero dimention.")
    else:
        b = torch.rand(b_size, device="cuda") > 0.5

    elapsed_time = benchmark_function(
        torch.ops.aten.masked_fill_, args=(a, b, 0), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for masked_fill_ (input size {size}): {elapsed_time * 10**6:.3f}"
    )


def benchmark_sub(size, n_warmup, n_iter):
    a_size = size[0]
    b_size = size[1]
    if len(a_size) == 0:
        raise ValueError("The input tensor cannot be zero dimention.")
    else:
        a = torch.rand(a_size, device="cuda")

    if len(b_size) == 0:
        b = 0
    else:
        b = torch.rand(b_size, device="cuda")

    elapsed_time = benchmark_function(
        torch.ops.aten.sub, args=(a, b, 1), n_warmup=n_warmup, n_iter=n_iter
    )

    print(
        f"Elapsed time (in us) for sub (input size {size}): {elapsed_time * 10**6:.3f}"
    )


# =====================
ops_to_fun = {
    "aten.bitwise_and": benchmark_bitwise_and,
    "aten.bitwise_or": benchmark_bitwise_or,
    "aten.div": benchmark_div,
    "aten.div_": benchmark_div_,
    "aten.eq": benchmark_eq,
    "aten.exponential_": benchmark_exponential_,
    "aten.fill_": benchmark_fill_,
    "aten.ge": benchmark_ge,
    "aten.index": benchmark_index,
    "aten.log": benchmark_log,
    "aten.lt": benchmark_lt,
    "aten.masked_fill_": benchmark_masked_fill_,
    "aten.sub": benchmark_sub,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", required=True)
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=20)
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        content = yaml.safe_load(file)
        for workload in content:
            print("=" * 15)
            print(workload["name"])
            for operator in workload["operators"]:
                func_to_call = ops_to_fun[operator["operator_name"]]
                size = eval(operator["size"])
                if "dtype" in operator:
                    dtype = getattr(torch, operator["dtype"].split(".")[-1])
                    func_to_call(size, args.n_warmup, args.n_iter, dtype=dtype)
                else:
                    func_to_call(size, args.n_warmup, args.n_iter)
