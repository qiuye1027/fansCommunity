# coding：utf-8
import random
import requests
import json
import time
from urllib.request import Request, urlopen
import ssl
import time
import hashlib

ssl._create_default_https_context = ssl._create_unverified_context


def getHoney():
    zz = {}
    now = round(time.time())

    e = hex(int(now)).upper()[2:]  # hex()转换一个整数对象为十六进制的字符串表示

    i = hashlib.md5(str(int(now)).encode('utf8')).hexdigest().upper()  # hashlib.md5().hexdigest()创建hash对象并返回16进制结果
    if len(e) != 8:
        zz = {'as': "479BB4B7254C150",
              'cp': "7E0AC8874BB0985"}
        return zz
    n = i[:5]
    a = i[-5:]
    r = ""
    s = ""
    for i in range(5):
        s = s + n[i] + e[i]
    for j in range(5):
        r = r + e[j + 3] + a[j]
    zz = {
        'as': "A1" + s + e[-3:],
        'cp': e[0:3] + r + "E1"
    }

    return  zz




firefox_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    "Cookie": "tt_webid=6665596295203948035; WEATHER_CITY=%E5%8C%97%E4%BA%AC; tt_webid=6665596295203948035; __tasessionId=7tzabdb541551957387330",
    "tt-request-time": str(int(time.time() * 1000)),
    "Referer": "http://download.google.com/ "
}



# url = ' https://www.toutiao.com/api/pc/feed/'

# 中国经济网
# url_ZGJJW = 'https://www.toutiao.com/c/user/article/?page_type=1&user_id=50502346296&max_behot_time=0&count=20&as=A1256CF73FBAE3A&cp=5C7FAA1EF37AFE1&_signature=wY8QYxAWnfp--eGFzu.KTsGPEH'
#
# max_behot_time =0
# max_behot_time_tmp = 0
# ass = getHoney()['as']
# cp = getHoney()['cp']

for index in range(500):

    time.sleep(5)
    print(index)
    url_TT = 'https://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1&max_behot_time=0&max_behot_time_tmp=1551957389&tadrequire=true&as=A1350C88A09FDF6&cp=5C801F1D5F760E1&_signature=NfDJfQAAaXaKhjiboxhFojXwyW'

    request = Request( url_TT, headers=firefox_headers  )
    html = urlopen( request ).read()
    strs = json.loads(html)

    # timss = strs["next"]["max_behot_time"]
    #
    # max_behot_time = timss
    # max_behot_time_tmp = timss

    for content in strs["data"]:
        print(content["title"])










