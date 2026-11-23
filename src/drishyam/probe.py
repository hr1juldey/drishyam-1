import sys

import torch


def main() -> None:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA device")
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    t = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda") * 2
    print(f"device={name} sm={cap[0]}{cap[1]} bf16_ok={t.item() == 2.0}")
    if cap < (8, 0):
        sys.exit("FAIL: bf16 needs SM >= 8.0")
    print("PROBE PASS")


if __name__ == "__main__":
    main()
