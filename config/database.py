import mysql.connector
from config import config
import re
import json

def insertData(jsonData):



    conn = mysql.connector.connect(
        user=config.DB_CONFIG['NAME'],
        password=config.DB_CONFIG['PASSWORD'],
        database='screenDB',
        use_unicode=True,

    )


    cursor = conn.cursor()

    #
    if table_exists(cursor, 'crawlerDataBG') == 0:

        cursor.execute(
        'CREATE TABLE `screenDB`.`crawlerDataBG` ( `index` int(255) NOT NULL AUTO_INCREMENT, `author` varchar(255), `putTime` date,`url` varchar(255), `title` varchar(255)  CHARACTER SET utf8, `second_title` varchar(255) CHARACTER SET utf8, PRIMARY KEY (`index`)) COMMENT=""')


    cursor.execute('insert into `screenDB`.`crawlerDataBG` (author, putTime,url, title, second_title) values (%s, %s, %s, %s,%s)', [jsonData['author'], jsonData['putTime'], jsonData['url'], jsonData['title'], jsonData['desc']])
    #
    # # cursor.execute(
    # #     'CREATE TABLE `screenDB`.`crawlerData` ( `index` int(255) NOT NULL AUTO_INCREMENT, `user` varchar(255), `source` varchar(255), `useful` varchar(255), `type` varchar(255), `comment` varchar(255) CHARACTER SET utf8, PRIMARY KEY (`index`)) COMMENT=""')
    # #
    # # cursor.execute(
    # #     'insert into `screenDB`.`crawlerData` (user, source, useful, type, comment) values (%s, %s, %s, %s, %s)',
    # #     [jsonData['user'], jsonData['source'], jsonData['useful'], jsonData['type'], jsonData['comment']])
    #
    conn.commit()
    cursor.close()
    conn.close()

def table_exists(con,table_name):
    sql = "show tables;"
    con.execute(sql)
    tables = [con.fetchall()]
    table_list = re.findall('(\'.*?\')',str(tables))
    table_list = [re.sub("'",'',each) for each in table_list]
    if table_name in table_list:
        return 1
    else:
        return 0

# insertData({'user': 'kelle001', 'source': '比亚迪F3论坛', 'useful': 1034, 'type': '好评', 'comment': '油耗及性价比空间还有操控绝对满意'})