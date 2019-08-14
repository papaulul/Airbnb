# -*- coding: utf-8 -*-
import scrapy
import csv
import time
from Airbnb.items import AirbnbItem
import numpy as np
class AirspiderSpider(scrapy.Spider):
    name = 'airspider'
    allowed_domains = ['airbnb.com']
    temp = []
    #with open('../files/listings_LA_7_8.csv') as csvfile:
    with open('../files/listings_SF_7_8.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter = ',')
        for row in readCSV:
            temp.append(row[1])
    start_urls = temp[1:]

    def parse(self, response):
        if len(response.xpath('//div[@class="_gor68n"]//title/text()').extract()) > 0:
            air = AirbnbItem()
            air['url'] = response.url
            if len(response.xpath('//span[@class="_13fmes0l"]/text()').extract()) == 1:
                air['title'] = response.xpath('//span[@class="_13fmes0l"]/text()').extract()[0]
            else: 
                air['title'] = np.NAN
            air['name'] = " ".join(response.xpath('//div[@class="_tw4pe52"]/text()').extract()[0].split(" ")[2:])
            yield air