import torch


def GetFaultLayerTypes():

    faultLayerTypes=[]
    faultLayerTypes.append('Conv2d')
    # faultLayerTypes.append('AdaptiveAvgPool2d')
    # faultLayerTypes.append('ReLU')
    return faultLayerTypes

def IsFaultLayer(className):
    faultLayerTypes=GetFaultLayerTypes()
    for t in faultLayerTypes:
        if t in className:
            return t
    else:
        return None

def GetBitErrorRate():
    ber=1e-7
    return ber

def GetDType():
    dtype=torch.float32
    return dtype

def GetDTypeName(dataType):
    table={
        torch.float16:'float16',
        torch.float32:'float32',
        torch.float64:'float64',
    }
    assert dataType  in table.keys()
    return table[dataType]

def GetSupportFloatDTypes():
    dtypes=[torch.float16,torch.float32,torch.float64]
    return dtypes

def IsSupportDType():
    if GetDType() in GetSupportFloatDTypes():
        return True
    else:
        return False

def Device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device