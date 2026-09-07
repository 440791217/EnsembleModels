import torch


def GetFaultLayerTypes():

    faultLayerTypes=[]
    faultLayerTypes.append('Conv')
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
        torch.qint8:'qint8',
        torch.quint8:'quint8',
    }
    assert dataType  in table.keys()
    return table[dataType]





def Device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device