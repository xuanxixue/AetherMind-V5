import sys
print("PYTHON_VERSION=" + sys.version.split()[0])
try:
    import torch
    print("TORCH=" + torch.__version__)
    print("CUDA_AVAILABLE=" + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA_VERSION=" + str(torch.version.cuda))
        print("DEVICE=" + torch.cuda.get_device_name(0))
        print("VCOUNT=" + str(torch.cuda.device_count()))
except Exception as e:
    print("TORCH_ERROR=" + str(e))