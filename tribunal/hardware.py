"""What machine am I on? One flat dict, printed after every run and written into the results CSV."""

import platform
import subprocess

import psutil
import torch


def _cpu_name():
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def describe(device, dtype):
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30),
        "torch": torch.__version__,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
    }
    if torch.cuda.is_available():
        props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        # ROCm builds also report as "cuda"; torch.version.hip is the tell.
        info["gpu_vendor"] = "AMD" if torch.version.hip else "NVIDIA"
        info["gpu"] = props[0].name
        info["gpu_count"] = len(props)
        info["gpu_mem_gb"] = round(props[0].total_memory / 2**30)
    return info


def machine_slug(info):
    """Short, hostname-free name for results files, e.g. 'apple-m2-pro' or 'nvidia-a100-sxm4-80gb'."""
    name = info.get("gpu", info["cpu"])
    return "-".join(name.lower().replace("(r)", "").replace("(tm)", "").split())
