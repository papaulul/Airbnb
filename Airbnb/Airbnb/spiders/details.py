# -*- coding: utf-8 -*-
import scrapy
import json
from Airbnb.items import AirbnbItem
import pandas as pd

class DetailsSpider(scrapy.Spider):
    name = 'details'
    allowed_domains = ['airbnb.com']
    # Copy and paste output of Airspider.py to start urls 
    files = pd.read_csv('plus-links.csv')
    current_list = list(map(lambda x: x[1:-1], files['url'].values))
    start_urls= list(set(current_list))
    

    def parse(self, response):
        #note: run multiple times. sometimes scrapy goes too fast and can't read so you will need to compile all times you ran the code and filter those that are duplicates
        air = AirbnbItem()
        # looks for host name
        air['name'] = response.xpath('//a[@href="#meet-your-host"]/text()').extract()
        # Sometimes listing title are located in different places, so I'm trying both methods 
        if response.xpath('//div[@style="margin-top:32px"]/span/text()').extract() is None:
            air['title']   = response.xpath('//span[@class="_1fkbblcp"]/text()').extract()
        else:
            air['title'] = response.xpath('//div[@style="margin-top:32px"]/span/text()').extract()
        air['url'] = response.url
        yield air
