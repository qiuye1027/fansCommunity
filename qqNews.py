import random
import requests
import json
import time
from urllib.request import Request, urlopen
import ssl
import time
import hashlib

ssl._create_default_https_context = ssl._create_unverified_context





firefox_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    # "Cookie": "tt_webid=6665596295203948035; WEATHER_CITY=%E5%8C%97%E4%BA%AC; tt_webid=6665596295203948035; __tasessionId=7tzabdb541551957387330",
    # "tt-request-time": str(int(time.time() * 1000)),
    "Referer": "https://news.qq.com/ "
}




for index in range(500):


    print(index)
    url_TT = 'https://pacaio.match.qq.com/irs/rcd?cid=108&ext=&token=349ee24cdf9327a050ddad8c166bd3e3&page=8&expIds=20190308004978|20190308A0JXLG|20190308004870|20190307000960|20190308A0NJ4W|20190308A0MVHJ|20190308A0D3JR|20190308A0HKG0|20181115002262|20190308A0I5HU|20190308A0QEW7|20190308004967|20190306A1HBB7|20190308A09ZEW|20190307005065&callback=__jp4'

    request = Request( url_TT, headers=firefox_headers  )
    html = urlopen( request ).read()
    strs = json.loads(html)
    print(strs)


    # for content in strs["data"]:
    #     print(content["title"])






