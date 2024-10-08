import sys
import torch
import time
import numpy as np
from pathlib import Path
import os
import logging

# (JPVILLAM): Removing for easy module porting
# from .megatron.model import LayerNorm
# from .megatron.model.fused_softmax import FusedScaleMaskSoftmax, SoftmaxFusionTypes
# from .megatron.model.transformer import ParallelSelfAttention, ParallelMLP, ParallelTransformerLayer
# from .megatron.model.transformer import bias_dropout_add_fused_train
# from .megatron.model.activations import bias_gelu_impl
# from .megatron.model.gpt2_model import gpt2_attention_mask_func as attention_mask_func
# from .megatron.model.word_embeddings import Embedding

from contextlib import contextmanager, nullcontext

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class Tee(object):
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        if self.verbose:
            self.stdout.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def min_max_avg(target):
    return [np.min(target), np.max(target), np.average(target)]


def display(shape):
    return "x".join([str(dim) for dim in shape])


@contextmanager
def torch_profiler_context(trace_file_name):
    trace_file_name = f"{trace_file_name}_torch.json"
    profile_dir = os.getenv("PROFILE_DIR", default="")
    os.makedirs(profile_dir, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    )
    profiler.start()
    try:
        with torch.no_grad():
            yield profiler
    finally:
        profiler.stop()
        print(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )
        profiler.export_chrome_trace(os.path.join(profile_dir, trace_file_name))


@contextmanager
def rpd_profiler_context(trace_file_name):
    from rpdTracerControl import rpdTracerControl

    trace_file_name = f"{trace_file_name}_torch.rpd"

    profile_dir = os.getenv("PROFILE_DIR", default="")
    os.makedirs(profile_dir, exist_ok=True)
    trace_file_path = os.path.join(profile_dir, f"{trace_file_name}_torch.rpd")
    with rpdTracerControl(trace_file_path, nvtx=True) as p:
        yield
    p.top_totals()


def get_profiling_context(profiling_mode, trace_file_name=None):
    if profiling_mode == "torch":
        return torch_profiler_context(trace_file_name)
    elif profiling_mode == "rpd":
        return rpd_profiler_context(trace_file_name)
    else:
        return nullcontext()


def benchmark_convolution(
    input_channels,
    output_channels,
    batch_size,
    input_size,
    kernel_size,
    stride,
    padding,
    num_iterations,
    num_warmup_iterations,
    profiling_mode="None",
):
    input_tensor = torch.randn(
        batch_size, input_channels, input_size, input_size, dtype=torch.float16
    ).to(device="cuda", memory_format=torch.channels_last)
    kernel = torch.randn(
        output_channels,
        input_channels,
        kernel_size,
        kernel_size,
        dtype=torch.float16,
        device="cuda",
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = np.zeros(num_iterations + num_warmup_iterations)
    if profiling_mode:
        trace_file_name = (
            f"conv_trace_{batch_size}_i{input_size}_k{kernel_size}_s{stride}_p{padding}"
        )
    else:
        trace_file_name = None

    with get_profiling_context(profiling_mode, trace_file_name):
        for i in range(num_warmup_iterations + num_iterations):
            start.record()
            output = torch.nn.functional.conv2d(
                input_tensor, kernel, stride=stride, padding=padding
            )
            end.record()
            torch.cuda.synchronize()
            times[i] = start.elapsed_time(end)

    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times) / 1000  # Convert to seconds

    # Calculate FLOPs for convolution
    input_size_after_padding = input_size + 2 * padding
    output_size = (input_size_after_padding - kernel_size) // stride + 1
    num_ops = 2 * (
        batch_size * output_channels * input_channels * kernel_size**2 * output_size**2
    )
    flops = num_ops / (10**12)  # FLOPs in TFLOPs
    throughput = flops / elapsed_time  # Throughput in TFLOP/s

    # Memory access calculations
    bytes_input = input_tensor.element_size() * input_tensor.nelement()
    bytes_kernel = kernel.element_size() * kernel.nelement()
    bytes_output = output.element_size() * output.nelement()
    total_bytes_accessed = bytes_input + bytes_kernel + bytes_output
    bandwidth = total_bytes_accessed / (elapsed_time * 10**9)  # Bandwidth in GB/s
    arithmetic_intensity = num_ops / total_bytes_accessed  # FLOPs/B
    print(f"conv2d: input_tensor: {input_tensor.shape}, Kernel: {kernel.shape}")
    print(
        f"batch {batch_size}, {input_size}x{input_size} input, {kernel_size}x{kernel_size} kernel, stride {stride}, padding {padding}"
    )
    print(f"input_channel {input_channels}, output_channel {output_channels}")
    print(f"Elapsed time (in us): {elapsed_time * 10**6:.3f}")
    print(f"Throughput (in TFLOP/s): {throughput:.3f}")
    print(f"Bandwidth (in GB/s): {bandwidth:.3f}")
    print(f"Arithmetic Intensity (in FLOPs/B): {arithmetic_intensity:.3f}")
    print("-" * 80)

    return [elapsed_time, throughput, bandwidth, arithmetic_intensity]


def get_torch_mm_func(dims):
    if dims == 2:
        return torch.mm
    elif dims == 3:
        return torch.bmm
    else:
        raise ValueError(f"Unsupported dim: {dims}")


TORCH_DTYPES = {
    "fp16": torch.half,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def generate_tensor(b, x, y, device=0, dtype="fp16"):
    assert dtype in TORCH_DTYPES, f"{dtype} not currently supported."
    size = (x, y) if b == 1 else (b, x, y)
    # XXX: Apparently the double "to" is faster
    return torch.randn(size).to(dtype=TORCH_DTYPES[dtype]).to(f"cuda:{device}")


# Benchmark of a basic GEMM
def benchmark_mm(
    m,
    n,
    k,
    num_iterations,
    num_warmup_iterations,
    b=1,
    dtype="fp16",
    transpose="TN",
    profiling=False,
    device="0",
):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # XXX: Note that if m or k = 1 and b = 0 pytorch will not do the transpose?
    if transpose[1] == "T":
        A = generate_tensor(b, k, n, device=device, dtype=dtype).transpose(-1, -2)
    else:
        A = generate_tensor(b, n, k, device=device, dtype=dtype)

    if transpose[0] == "T":
        B = generate_tensor(b, m, k, device=device, dtype=dtype).transpose(-1, -2)
    else:
        B = generate_tensor(b, k, m, device=device, dtype=dtype)
    times = np.zeros(num_iterations + num_warmup_iterations)
    cpu_times = np.zeros(num_iterations + num_warmup_iterations)

    torch.cuda.set_device(f"cuda:{device}")

    profile_dir = os.getenv("PROFILE_DIR", default="")
    trace_file_name = f"{profile_dir}gemm_trace_m{m}_n{n}_k{k}.json"

    mm = get_torch_mm_func(A.dim())

    with get_profiling_context("torch" if profiling else "", trace_file_name):
        for i in range(num_warmup_iterations + num_iterations):
            start_time = time.perf_counter()
            # with torch.no_grad():
            start.record()
            mm(A, B)
            end.record()
            torch.cuda.synchronize()
            cpu_times[i] = time.perf_counter() - start_time
            times[i] = start.elapsed_time(end)

    times = times[num_warmup_iterations:] / 1000
    cpu_times = cpu_times[num_warmup_iterations:]
    overhead = cpu_times - times
    overhead_p = 1 - (times / cpu_times)
    overhead_perc = min_max_avg(overhead_p)
    overhead_time = min_max_avg(overhead)
    elapsed_time = min_max_avg(times)
    elapsed_cpu_time = min_max_avg(cpu_times)

    FLOP = 2 * m * n * k
    MEM_access = A.element_size() * (m * k + n * k + m * n)
    bandwidth = MEM_access / (elapsed_time[2] * 10**9)
    arithmetic_intensity = FLOP / MEM_access

    throughput = FLOP / (elapsed_time[2] * 10**12)
    log.debug(f"Results for GEMM {m}x{n}x{k}")
    log.debug(
        f"Elapsed GPU time (in us) min/max/avg: {elapsed_time[0] * 10**6:.3f} {elapsed_time[1] * 10**6:.3f} {elapsed_time[2] * 10**6:.3f}"
    )
    log.debug(
        f"Elapsed CPU time (in us)min/max/avg: {elapsed_cpu_time[0] * 10**6:.3f} {elapsed_cpu_time[1] * 10**6:.3f} {elapsed_cpu_time[2] * 10**6:.3f}"
    )
    log.debug(
        f"CPU overhead (in us) min/max/avg: {overhead_time[0] * 10**6:.3f} {overhead_time[1] * 10**6:.3f} {overhead_time[2] * 10**6:.3f}"
    )
    log.debug(
        f"CPU overhead (%) min/max/avg: {overhead_perc[0] * 100:.1f} {overhead_perc[1] * 100 :.1f} {overhead_perc[2] * 100:.1f}"
    )
    log.debug(f"Throughput (in TFLOP/s): {throughput:.3f}")
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem
    log.debug(f"Bandwidth (in GB/s): {bandwidth:.3f}")
    log.debug(f"Arithmetic Intensity (in FLOPs/B): {arithmetic_intensity:.3f}")
    log.debug("-" * 80)
    return [
        elapsed_time,
        elapsed_cpu_time,
        overhead_time,
        overhead_perc,
        throughput,
        bandwidth,
        arithmetic_intensity,
    ]


# Benchmark of a GEMM with a single batched operator
def benchmark_mm_b(m, n, k, label, b, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    B = torch.randn((k, n)).half().to("cuda")
    if b is None:
        A = torch.randn((m, n)).half().to("cuda")
        C = torch.empty((m, k)).half().to("cuda")
        b = 1
    else:
        A = torch.randn((b, m, n)).half().to("cuda")
        C = torch.empty((b, m, k)).half().to("cuda")
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.nn.functional.linear(A, B, out=C)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    FLOP = 2 * m * n * k * b
    MEM_access = A.element_size() * (m * k + n * k + m * n) * b
    print(
        f"Elapsed time (in us) for {label} ({m}x{n}x{k}, b={b}): {elapsed_time * 10**6 :.3f}"
    )
    print(
        f"Throughput (in TFLOP/s) for {label} ({m}x{n}x{k}, b={b}): "
        f"{FLOP / (elapsed_time * 10**12):.3f}"
    )
    print(
        f"Bandwidth (in GB/s) for {m}x{n}x{k}, b={b}: {(MEM_access / (elapsed_time * 10**9)):.3f}"
    )
    print(
        f"Arithmetic Intensity (in FLOPs/B) for {m}x{n}x{k}, b={b}: {(FLOP/MEM_access):.3f}"
    )
    return elapsed_time


def benchmark_bmm(b, m, n, k, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn((b, m, n)).half().to("cuda")
    B = torch.randn((b, n, k)).half().to("cuda")
    C = torch.empty((b, m, k)).half().to("cuda")
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.bmm(A, B, out=C)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    FLOP = 2 * m * n * k * b
    MEM_access = A.element_size() * (m * k + n * k + m * n) * b
    print(
        f"Elapsed time (in us) for {label} ({b}x{m}x{n}x{k}): {elapsed_time * 10**6:.3f}"
    )
    print(
        f"Throughput (in TFLOP/s) for {label} ({b}x{m}x{n}x{k}): "
        f"{FLOP / (elapsed_time * 10**12):.3f}"
    )
    print(
        f"Bandwidth (in GB/s) for {b}x{m}x{n}x{k}: {(MEM_access / (elapsed_time * 10**9)):.3f}"
    )
    print(
        f"Arithmetic Intensity (in FLOPs/B) for {b}x{m}x{n}x{k}: {(FLOP/MEM_access):.3f}"
    )
    return elapsed_time


def benchmark_dropout(A_dim, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    dropout = torch.nn.Dropout(0.5).to("cuda")

    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            dropout(A)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time


def benchmark_softmax(
    scores_shape, seq_length, label, num_iterations, num_warmup_iterations
):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    scores = torch.randn(scores_shape).half().to("cuda")
    attention_mask = torch.tril(
        torch.ones((1, seq_length, seq_length), device="cuda")
    ).view(1, 1, seq_length, seq_length)
    attention_mask = attention_mask < 0.5
    softmax = FusedScaleMaskSoftmax(
        True,
        False,
        SoftmaxFusionTypes.none,  # attentionmasktype.padding=1,True
        attention_mask_func,
        True,
        1,
    )
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            softmax(scores, attention_mask)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    print(f"Elapsed time for {label} ({display(scores_shape)}): {elapsed_time :.4f}")
    return elapsed_time


def benchmark_fused_gelu(A_dim, b_dim, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    b = torch.randn(b_dim).half().to("cuda")
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_gelu_impl(A, b)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time


def benchmark_layer_norm(
    A_dim, normalized_shape, label, num_iterations, num_warmup_iterations
):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(A_dim).half().to("cuda")
    layer_norm = LayerNorm(normalized_shape).half().to("cuda")
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            layer_norm(A)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    print(f"Elapsed time for {label} ({display(A_dim)}): {elapsed_time :.4f}")
    return elapsed_time


def benchmark_add_bias_dropout(shape, label, num_iterations, num_warmup_iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    A = torch.randn(shape).half().to("cuda")
    bias = torch.randn(shape).half().to("cuda")
    residue = torch.randn(shape).half().to("cuda")
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            bias_dropout_add_fused_train(A, bias, residue, 0.0)
            end.record()
        torch.cuda.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amax(times) / 1000
    print(f"Elapsed time for {label} ({display(shape)}): {elapsed_time :.4f}")
    return elapsed_time
