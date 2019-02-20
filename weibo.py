import urllib
import urllib.request
from bs4 import BeautifulSoup
import ssl
from config import config
from config import database
from config import utils

# 设置目标url，使用urllib.request.Request创建请求
url0 = "https://s.weibo.com/top/summary?cate=total&key=person"
req0 = urllib.request.Request(url0)


# 使用add_header设置请求头，将代码伪装成浏览器
req0.add_header("User-Agent",'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36')
ssl._create_default_https_context = ssl._create_unverified_context
# 使用urllib.request.urlopen打开页面，使用read方法保存html代码
html0 = urllib.request.urlopen(req0).read()

# 使用BeautifulSoup创建html代码的BeautifulSoup实例，存为soup0
soup0 = BeautifulSoup(html0)

total = soup0.find("div", id="pl_top_realtimehot").find("tbody").findAll("tr")

item = {}
for i in list(total):

     ins = total.index(i)

     try:

            ss = "正在爬取第%d条热点" % (ins)
            print(ss)


            try:
                rank = i.find("td", class_="ranktop").get_text()
            except:
                rank = ""

            try:
                title = i.find("td", class_="td-02").find("a").get_text()
            except:
                title = ""

            try:
                href = i.find("td", class_="td-02").find("a")["href"]
            except:
                href = ""

            try:
                number = i.find("td", class_="td-02").find("span").get_text()
            except:
                number = ""

            item["rank"] = rank
            item["href"] = href
            item["title"] = title
            item["number"] = number

            database.insertDataWeibo(item)

     except:
            s = "爬取第%d条热点失败" % (ins)
            print(s)
            pass
