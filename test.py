import urllib
import urllib.request
from bs4 import BeautifulSoup
import re
import random
import time
from config import config
from config import database
from config import utils
import json
import os
import datetime

# print(config.DB_CONFIG['DB_CONNECT_STRING'])
# print(config.get_header())


# 设置目标url，使用urllib.request.Request创建请求
url0 = "http://newcar.xcar.com.cn/257/review/0.htm"
req0 = urllib.request.Request(url0)

# 使用add_header设置请求头，将代码伪装成浏览器
req0.add_header("User-Agent", config.USER_AGENTS[5])

# 使用urllib.request.urlopen打开页面，使用read方法保存html代码
html0 = urllib.request.urlopen(req0).read()

# 使用BeautifulSoup创建html代码的BeautifulSoup实例，存为soup0
soup0 = BeautifulSoup(html0)

# 获取尾页（对照前一小节获取尾页的内容看你就明白了）
total_page = int(soup0.find("div", class_="pagers").findAll("a")[-2].get_text())
myfile = open("./dist/aika_qc_gn_1_1_1.txt", "a")
print("user", "来源", "认为有用人数", "类型", "comment", sep="|", file=myfile)
ll = 0
arr = []

for i in list(range(1, total_page + 1)):

    # 设置随机暂停时间
    stop = random.uniform(1, 3)
    url = "http://newcar.xcar.com.cn/257/review/0/0_" + str(i) + ".htm"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", config.USER_AGENTS[5])
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html)
    contents = soup.find('div', class_="review_comments").findAll("dl")

    l = len(contents)

    item = {}

    for content in contents:
        ll = ll + 1

        if i > 2:
            break

        tiaoshu = contents.index(content)
        try:
            ss = "正在爬取第%d页的第%d的评论，网址为%s" % (i, tiaoshu + 1, url)
            print(ss)
            try:
                comment_jiaodu = content.find("dt").find("em").find("a").get_text()
            except:
                comment_jiaodu = ""

            try:
                comment_type0 = content.find("dt").get_text()
                comment_type1 = comment_type0.split("【")[1]
                comment_type = comment_type1.split("】")[0]
            except:
                comment_type = "好评"

            # 认为该条评价有用的人数
            try:
                useful = int(content.find("dd").find("div", class_="useful").find("i").find("span").get_text())
            except:
                useful = ""

            # 评论来源
            try:
                comment_region = content.find("dd").find("p").find("a").get_text()
            except:
                comment_region = ""

            # 评论者名称
            try:
                user = content.find("dd").find("p").get_text().split("：")[-1]
            except:
                user = ""

            # 评论内容
            try:
                comment_url = content.find('dt').findAll('a')[-1]['href']
                urlc = comment_url
                reqc = urllib.request.Request(urlc)
                reqc.add_header("User-Agent", config.USER_AGENTS[5])
                htmlc = urllib.request.urlopen(reqc).read()
                soupc = BeautifulSoup(htmlc)
                comment0 = \
                    soupc.find('div', id='mainNew').find('div', class_='maintable').findAll('form')[1].find('table', class_='t_msg').findAll('tr')[1]
                try:
                    comment = comment0.find('font').get_text().strip().replace("\n", "").replace("\t", "")
                except:
                    comment = ""
                    # try:
                    #     comment_time = soupc.find('div', id='mainNew').find('div', class_='maintable').findAll('form')[1].find('table', class_='t_msg').\
                    #     find('div', style='padding-top: 4px;float:left').get_text()
                    # except:
                    #     comment_time = ""
            except:
                try:
                    comment = content.find("dd").get_text().split("\n")[-1].split('\r')[-1].split("：")[-1]
                except:
                    comment = ""

            # time.sleep(stop)
            item["user"] = user
            item["source"] = comment_region
            item["useful"] = useful
            item["type"] = comment_type
            item["comment"] = comment

            arr.append(json.dumps(item, ensure_ascii=False))
            strval = ','.join(arr)

            date = datetime.datetime.now().strftime('%Y_%m_%d')  # 2019_02_13
            fileUrl = os.getcwd() + "/dist/data/"  ##获取此py文件路径
            utils.txt(fileUrl, date, strval)
            # print(item)
            database.insertData(item)
            print(user, comment_region, useful, comment_type, comment, sep="|", file=myfile)



        except:
            s = "爬取第%d页的第%d的评论失败，网址为%s" % (i, tiaoshu + 1, url)
            print(s)
            pass

myfile.close()
