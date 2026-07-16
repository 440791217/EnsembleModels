import numpy as np
import torch


def Bitflip(x, ber):
    """
    对 tensor 的底层二进制位按 BER 独立翻转。
    支持 float64/float32/float16/int16/int8 等。
    """
    x = x.contiguous()
    y = x.clone()

    # 把底层存储按 uint8 字节查看
    y_bytes = y.view(torch.uint8)
    # print(y_bytes.shape)

    # 每个字节 8 个 bit，每个 bit 独立 Bernoulli(ber)
    bit_error = torch.bernoulli(
        torch.full((*y_bytes.shape, 8), ber, device=y.device)
    ).to(torch.uint8)

    bit_values = (2 ** torch.arange(8, device=y.device)).to(torch.uint8)

    # 每个字节对应一个 0~255 的翻转掩码
    byte_flip_mask = (bit_error * bit_values).sum(dim=-1).to(torch.uint8)

    # XOR 位翻转
    y_bytes ^= byte_flip_mask

    return y

def Test():
    x16=torch.ones((1, 4, 1, 1), dtype=torch.float16)
    x32=torch.ones((1, 4, 1, 1), dtype=torch.float32)
    print(x16)
    print(x32)
    # x_int8 = torch.ones((1, 64, 32, 32), dtype=torch.int8)
    # y_int8 = InjectBerBitflip(x_int8, ber=1e-5)

if __name__=='__main__':
    Test()

