from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from pyquery import PyQuery as pq

import pymysql

browser = webdriver.Chrome()
wait = WebDriverWait(browser, 10)
# 进入爬取页面
def search():
    try:
        url = 'http://smart.huanqiu.com/travel/'
        browser.get(url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'boxCon')))
        getDetail()
        total = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.pageBox > div > a:nth-child(13) ')))
        return total.text
    except TimeoutError:
        return search()
# 得到具体信息
def getDetail():
    html = pq(browser.page_source, parser="html")
    content = html.find('.boxCon > .fallsFlow')
    uls = content.find('ul').items()

    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="123456", db="datamining",
                          charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    for ul in uls:
        lis = ul('li').items()
        for li in lis:
            news = {
                'title': li.find('h3 > a').text(),
                'href': li.find('a').attr('href'),
                'time': li.find('h6').text()
            }
            try:
                temp = (news['title'], news['href'], news['time'],)
                sql = "insert into technology_news_new(title,href,time) values(%s,%s,%s)"
                cur.execute(sql, temp)
                cur.connection.commit()
                # print(news)
            except Exception as error:
                print(error)
    conn.close()

# 爬取下一页
def next_detail(page_number):
    try:
        nextBotton = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.pageBox > div > a:last-child')))
        nextBotton.click()
        wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, '.pageBox > div > em'), str(page_number)))
        getDetail()
    except TimeoutException:
        next_detail(page_number)

def main():
    try:
        total = search()
        total = int(total)
        print(total)
        for i in range(2, total + 1):
            next_detail(i)
    except Exception:
        print(Exception)
    finally:
        browser.close()

if __name__ == '__main__':
main()
