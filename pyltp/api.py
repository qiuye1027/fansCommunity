# web server : https://doc.xfyun.cn/rest_api/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%9F%BA%E7%A1%80%E5%A4%84%E7%90%86.html




# 分句
# 使用 pyltp 进行分句示例如下

# -*- coding: utf-8 -*-
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
print('\n'.join(sents))





# 分词
# 使用 pyltp 进行分词示例如下

# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
words = segmentor.segment('元芳你怎么看')  # 分词
words_list = list(words)
print(words_list)
print('\t'.join(words))
segmentor.release()  # 释放模型




# 使用分词外部词典
# pyltp 分词支持用户使用自定义词典。分词外部词典本身是一个文本文件（plain text），每行指定一个词，编码同样须为 UTF-8，样例如下所示

# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, '/path/to/your/lexicon') # 加载模型，第二个参数是您的外部词典文件路径
words = segmentor.segment('亚硝酸盐是一种化学物质')
print('\t'.join(words))
segmentor.release()



# 使用个性化分词模型
# https://ltp.readthedocs.io/zh_CN/latest/theory.html#customized-cws-reference-label
# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

from pyltp import CustomizedSegmentor
customized_segmentor = CustomizedSegmentor()  # 初始化实例
customized_segmentor.load(cws_model_path, '/path/to/your/customized_model') # 加载模型，第二个参数是您的增量模型路径
words = customized_segmentor.segment('亚硝酸盐是一种化学物质')
print('\t'.join(words))
customized_segmentor.release()



#
# 词性标注
# 使用 pyltp 进行词性标注示例如下
# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

from pyltp import Postagger
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']  # 分词结果
postags = postagger.postag(words)  # 词性标注

print('\t'.join(postags))
postagger.release()  # 释放模型



# 命名实体识别
# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
netags = recognizer.recognize(words, postags)  # 命名实体识别

print('\t'.join(netags))
recognizer.release()  # 释放模型



# 依存句法分析
# 使用 pyltp 进行依存句法分析示例如下


# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

from pyltp import Parser
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
arcs = parser.parse(words, postags)  # 句法分析

print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
parser.release()  # 释放模型




# 语义角色标注
# 使用 pyltp 进行语义角色标注示例如下

# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/path/to/your/ltp_data'  # ltp模型目录的路径
srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。

from pyltp import SementicRoleLabeller
labeller = SementicRoleLabeller() # 初始化实例
labeller.load(srl_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
# arcs 使用依存句法分析的结果
roles = labeller.label(words, postags, arcs)  # 语义角色标注

# 打印结果
for role in roles:
    print(role.index, "".join(
        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
labeller.release()  # 释放模型