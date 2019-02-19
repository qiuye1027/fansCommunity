import os

def txt(path, name , text):  # 定义函数名
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    xxoo = path + name + '.json'  # 在当前py文件所在路径下的new文件中创建json

    file = open(xxoo, 'w')

    file.write(text)  # 写入内容信息

    file.close()
    print('ok')

