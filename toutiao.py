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
    t = int(time.time() / 1e3)
    e = str(hex(t)).upper()
    i = hashlib.md5(str(t).encode('utf8')).hexdigest().upper()

    # e = '0X17AE8F'
    # i ='7668F44EF5FD3601B5A1F6F518561342'

    if 8 != len(e) :
        ass = "479BB4B7254C150"
        cp = "7E0AC8874BB0985"

    n = i[:5]
    a = i[:-5]
    s = ""
    r = ""
    for ins in range(5):
        s = s + n[ins] + e[ins]
        r = r + e[ins + 3] + a[ins]


    ass =  "A1" + s + e[:-3]
    cp =  e[:3] + r + "E1"

    return  ass,cp




firefox_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    "Cookie": "tt_webid=6664799372042290691; WEATHER_CITY=%E5%8C%97%E4%BA%AC; tt_webid=6664799372042290691; csrftoken=1fb0f06209814cc47b62932aab2d5717; __tasessionId=5prxdnhkn1551868099494",
    "tt-request-time": str(int(time.time() * 1000)),
    "Referer": "http://download.google.com/ "
}

 

# url = ' https://www.toutiao.com/api/pc/feed/'

# 中国经济网
# url_ZGJJW = 'https://www.toutiao.com/c/user/article/?page_type=1&user_id=50502346296&max_behot_time=0&count=20&as=A1256CF73FBAE3A&cp=5C7FAA1EF37AFE1&_signature=wY8QYxAWnfp--eGFzu.KTsGPEH'
#
max_behot_time =0
max_behot_time_tmp = 0
ass = getHoney()[0]
cp = getHoney()[1]

for index in range(500):


    url_TT = 'https://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1&max_behot_time='+str(max_behot_time)+'&max_behot_time_tmp='+str(max_behot_time_tmp)+'&tadrequire=true&as='+ass+'&cp='+cp+'&_signature=QgnuOgAAHnj9fx.c6c0vrkIJ7i'
    print(index)
    request = Request( url_TT, headers=firefox_headers  )
    html = urlopen( request ).read()
    strs = json.loads(html)
    timss = strs["next"]["max_behot_time"]

    max_behot_time = timss
    max_behot_time_tmp = timss

    for content in strs["data"]:
        print(content["title"])


