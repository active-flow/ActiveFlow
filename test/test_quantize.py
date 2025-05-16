import torch
from quantization.quantizer import Q40Quantizer


def print_tensor(name, tensor):
    print(f"{name}:")
    print(tensor.shape)
    print(tensor.stride())
    print(tensor)


def quantize_4bit_example():
    N = 4
    K = 64
    #A = torch.zeros([N, K], dtype=torch.float16, device=torch.device("cuda:0"))
    #for i in range(N):
    #    A.data[i] = 1
    torch.manual_seed(45)
    A = torch.rand([N, K], dtype=torch.float16, device=torch.device("cuda:0"))
    print_tensor("A", A)
    q_mod = Q40Quantizer(q_group_size=64)
    out = q_mod(A)
    print_tensor("out", out)
    #C = dequantize_4bit(out, state, blocksize=64, quant_type="fp4")
    #print_tensor("C", C)


if __name__ == "__main__":
    quantize_4bit_example()
