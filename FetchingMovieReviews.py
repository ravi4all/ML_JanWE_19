import bs4
import urllib.request as url

movieName = input("Enter movie name : ")
web = url.urlopen('https://www.imdb.com/find?ref_=nv_sr_fn&q='+movieName)

bs = bs4.BeautifulSoup(web, 'lxml')
td = bs.find('td', class_='result_text')
href = td.find('a')['href']
newUrl = 'https://www.imdb.com'+href
web = url.urlopen(newUrl)
bs = bs4.BeautifulSoup(web, 'lxml')
div = bs.find('div', class_='user-comments')
href = div.find_all('a')[4]['href']
newUrl = 'https://www.imdb.com'+href
page = url.urlopen(newUrl)
bs = bs4.BeautifulSoup(page, 'lxml')
titles = bs.find_all('a', class_='title')
# print(titles)
for title in titles:
    print(title.text)
