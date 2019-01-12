# -*- coding: utf-8 -*-
import sys
import re
import jieba.posseg
import os
import MySQLdb
reload(sys)
sys.setdefaultencoding('utf-8')

def savefile(dirpath,filepath,content):

    if os.path.exists(dirpath)==False:
        os.makedirs(dirpath)

    savepath = os.path.join(dirpath,filepath)
    print(savepath)
    with open(savepath,"w") as fp:
        fp.write(content)
    fp.close()

def function(datalist,path):
    stopwords = []
    for word in open('stop_words_ch.txt', 'r'):
        stopwords.append(word.strip())

    db = MySQLdb.connect("localhost", "root", "root", "datamining", charset='utf8')
    cursor = db.cursor()

    i = 200001

    for item in datalist:
        sql = "select content from " + item
        cursor.execute(sql)
        results = cursor.fetchall()

        for row in results:
            content = []
            temptest = re.sub("[@·《》、.%，。？“”（）：(\u3000)(\xa0)！… ；▼]|[a-zA-Z0-9]|['月''日''年']", "", row['content'])
            words = jieba.posseg.cut(temptest)
            for w in words:
                if w.word not in stopwords and w.flag == 'n':
                    content.append(w.word)

            forpath = path + "/" + item.replace("_even","").replace("_odd","")
            subpath = str(i) +".txt"
            savefile(forpath,subpath," ".join(content))
            i = i+1
    conn.close()

if __name__=="__main__":
    test = ['education_odd', 'ent_odd','food_odd', 'healthy_odd','history_odd', 'houseproperty_odd','military_odd', 'sport_odd','tock_odd', 'technology_odd']
    path = ""
    function(test, path)
    # train = ['education_even,ent_even,food_even,healthy_even,history_even,houseproperty_even,military_even,sport_even,stock_even,technology_even]
    # path = ""
    # function(train, path)