import torch



def Error(message):
    print(message)
    exit(-1)


if __name__=='__main__':
    Error("{} is not support".format(torch.float32))