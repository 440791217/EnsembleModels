import sys
print(sys.path)

###建立连接

while True:
    a=input('请输入命令：')
    if a=='1':
        print('执行命令1')
        ###发送指定命令
    elif a=='2':
        print('执行命令2')
    elif a=='111':
        ####断开连接
        print('结束')
        break
    else:
        print('无效命令什么都不执行')