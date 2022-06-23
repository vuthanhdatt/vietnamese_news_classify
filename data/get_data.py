import requests
import pandas as pd
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup



def get_titles(category, start_date, end_date):
  delta = timedelta(days=1)
  titles = []
  while start_date <= end_date:
      date = start_date.strftime("%d/%m/%Y")
      unix_time = int(datetime.strptime(date, '%d/%m/%Y').strftime("%s"))
      r = requests.get(f'https://vnexpress.net/category/day?cateid={category}&fromdate={unix_time}&todate={unix_time}&allcate={category}')
      soup = BeautifulSoup(r.text, 'html.parser')
      raw_titles = soup.find_all('h3','title-news')
      for title in raw_titles:
        titles.append(title.find('a')['title'])
      start_date += delta
  return titles


if __name__ == '__main__':

    titles = []
    cate = []
    start_date = date(2021, 6, 20)
    end_date = date(2022, 6, 20)
    category_id = {
        'Thế giới': '1001002',
        'Thời sự': '1001005',
        'Thể thao': '1002565',
        'Kinh doanh': '1003159',
        'Sức khỏe':'1003750',

    }

    for category, category_id in category_id.items():
        title = get_titles(category_id, start_date, end_date)
        titles.extend(title)
        cate.extend([category]*len(title))
    df = pd.DataFrame({'Title': titles, 'Category': cate})
    df.to_csv('data.csv', index=False, encoding='utf-8')