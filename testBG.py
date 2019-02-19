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
url0 = "http://www.bugutime.com/news?page=1"
req0 = urllib.request.Request(url0)

# 使用add_header设置请求头，将代码伪装成浏览器
req0.add_header("User-Agent", config.USER_AGENTS[5])

# 使用urllib.request.urlopen打开页面，使用read方法保存html代码
html0 = urllib.request.urlopen(req0).read()

# 使用BeautifulSoup创建html代码的BeautifulSoup实例，存为soup0
soup0 = BeautifulSoup(html0)

# 获取尾页（对照前一小节获取尾页的内容看你就明白了）
total_page = int(soup0.find("ul", class_="pagination").findAll("a")[-1]["data-page"])

arr = []

for i in list(range(1, total_page + 1)):

    # 设置随机暂停时间
    stop = random.uniform(1, 3)
    url = "http://www.bugutime.com/news?page=" + str(i)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", config.USER_AGENTS[5])
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html)
    contents = soup.find('ul', class_="article-list").findAll("li")

    l = len(contents)

    item = {}

    for content in contents:

        if i > 20:
            break

        tiaoshu = contents.index(content)
        url_item = content.find("a")["href"]
        try:

            ss = "正在爬取第%d页的第%d条新闻，网址为%s" % (i, tiaoshu + 1, url_item)
            print(ss)

            try:
                author = content.find("span", class_="item-author").find("a").get_text()
            except:
                author = ""

            try:
                time = content.find("time").get_text()
            except:
                time = ""

            try:
                title = content.find("h3", class_="item-title").find("a").get_text()
            except:
                title = ""

            try:
                desc = content.find("p", class_="item-desc").get_text()
            except:
                desc = ""



            item["author"] = author
            item["putTime"] = time
            item["title"] = title
            item["desc"] = desc
            item["url"] = url_item

            arr.append(json.dumps(item, ensure_ascii=False))
            strval = ','.join(arr)

            date = datetime.datetime.now().strftime('%Y_%m_%d')  # 2019_02_13
            fileUrl = os.getcwd() + "/dist/data/"  ##获取此py文件路径
            utils.txt(fileUrl, date, strval)
            # print(item)
            database.insertData(item)

        except:
            s = "爬取第%d页的第%d条新闻失败，网址为%s" % (i, tiaoshu + 1, url_item)
            print(s)
            pass


