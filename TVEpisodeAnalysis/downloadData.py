import numpy as np
import bs4 as bs
import requests




site = 'http://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=parks-and-recreation'


def get_data():

	sause = requests.get(site).text
	soup = bs.BeautifulSoup(sause)

	episode_urls = []

	for url in soup.find_all('a'):
		urls = url.get('href')
		if urls[0] == 'v':
			episode_urls.append(urls)

	print 'Starting to download episodes'
	for episode in episode_urls:
		print 'Downloading Episode...', episode[-6:]
		episode_url = 'http://www.springfieldspringfield.co.uk/' + episode
		sause = requests.get(episode_url).text
		soup = bs.BeautifulSoup(sause)

		raw_text = soup.find('div', class_ = 'scrolling-script-container').get_text()
		raw_text = raw_text.replace('\n', '').replace('\t', '').strip()
		raw_text = raw_text.encode('utf-8')
		with open('data/' + episode[-6:] + '.txt', 'w') as f:
			f.write(raw_text) 



if __name__ == '__main__':
	get_data()
