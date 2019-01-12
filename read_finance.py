
import pymysql
import requests
from lxml import etree


conn = pymysql.connect(host="127.0.0.1", user="root", passwd="123456", db="datamining",
                       charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()
sql = "select href from technology_news_new"
cur.execute(sql)

results = cur.fetchall()
for row in results:
    try:
        response = requests.get(row['href'])
        response.encoding = response.apparent_encoding
        root = etree.HTML(response.content.decode('utf-8', 'ignore'))

        content = ''.join(root.xpath("//div[@class='la_con']/p/text()"))
        #print(content)
        sql = "insert into technology(content) values(%s)"
        cur.execute(sql, content)
        cur.connection.commit()
    except:
        print("Error1")

conn.close()
