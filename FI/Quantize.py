import torch


def QuantizeInt8(x: torch.Tensor):
    max_val = x.abs().max()

    # 防止除0
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127

    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale

def QuantizeInt16(x: torch.Tensor):
    max_val = x.abs().max()

    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 32767

    x_q = (x / scale).round().clamp(-32768, 32767).to(torch.int16)
    return x_q, scale

def Dequantize(x_q: torch.Tensor, scale):
    return x_q.float() * scale


def Test():
    # 创建一个 Tensor
    tensor = torch.tensor([[-1, 2, 3], [4, 5, 8]])
    xQ, scale=QuantizeInt8(tensor)
    # 获取 Tensor 的形状（shape）
    print("Tensor shape:", tensor.shape)

    # 获取 Tensor 的数据类型（dtype）
    print("Tensor data type:", tensor.dtype)

    # 获取 Tensor 存储的设备（device）
    print("Tensor device:", tensor.device)
    print("xQ:{},scale:{}".format(xQ,scale)) 
    xQ=xQ.to(torch.float32)
    pass

if __name__=='__main__':
    Test()