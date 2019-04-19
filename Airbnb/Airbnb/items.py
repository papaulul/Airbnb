# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class AirbnbItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    url = scrapy.Field()
    url_num = scrapy.Field()
    name = scrapy.Field()
    title = scrapy.Field()

class ProductionItem(scrapy.Item):
    img_url = scrapy.Field()

class ImageItem(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()