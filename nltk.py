import fool

# text = "语言技术平台(Language Technology Platform，LTP)是 哈工大社会计算与信息检索研究中心 历时十年开发的一整套中文语言处理系统。LTP制定了基于XML的语言处理结果表示，并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块(包括词法、句法、语义等6项中文处理核心技术)，以及基于动态链接库(Dynamic Link Library, DLL)的应用程序接口，可视化工具，并且能够以网络服务(Web Service)的形式进行使用。"
# print(fool.cut(text))

map_file = "./flloNLTK/demo/maps.pkl"
checkpoint_ifle = "./results/demo_seg/modle.pb"

smodel = fool.load_model(map_file=map_file, model_file=checkpoint_ifle)
tags = smodel.predict(["北京欢迎你", "你在哪里"])
print(tags)