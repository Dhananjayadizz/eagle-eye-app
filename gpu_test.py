import torch

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"\n=== GPU Information ===")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        # Test tensor operations
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y
        print("\nMatrix multiplication test successful!")
        print(z)
    else:
        print("\nTroubleshooting:")
        print("1. Verify NVIDIA drivers with 'nvidia-smi'")
        print("2. Check CUDA version with 'nvcc --version'")
        print("3. Confirm PATH includes CUDA 12.6 binaries")

if __name__ == "__main__":
    main()

    