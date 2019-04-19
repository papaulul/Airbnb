# -*- coding: utf-8 -*-
import scrapy
from selenium import webdriver
import time
from Airbnb.items import AirbnbItem
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
import pandas as pd
import random
class AirspiderSpider(scrapy.Spider):
    name = 'Airspider'
    allowed_domains = ['airbnb.com']
    start_urls = ['https://www.airbnb.com/s/Los-Angeles--CA--United-States/plus_homes?refinement_paths%5B%5D=%2Fselect_homes&query=Los%20Angeles%2C%20CA%2C%20United%20States&adults=0&children=0&infants=0&guests=0&place_id=ChIJE9on3F3HwoAR9AhGJW_fL-I&click_referer=t%3ASEE_ALL%7Csid%3A5425de3b-fcab-4b2b-8f9d-06bcf808fe6e%7Cst%3AHOME_GROUPING_SELECT_HOMES&superhost=false&title_type=SELECT_GROUPING&allow_override%5B%5D=&s_tag=ShdJZVjc']
    #LA
    #'https://www.airbnb.com/s/Los-Angeles--CA--United-States/plus_homes?refinement_paths%5B%5D=%2Fselect_homes&query=Los%20Angeles%2C%20CA%2C%20United%20States&adults=0&children=0&infants=0&guests=0&place_id=ChIJE9on3F3HwoAR9AhGJW_fL-I&click_referer=t%3ASEE_ALL%7Csid%3A5425de3b-fcab-4b2b-8f9d-06bcf808fe6e%7Cst%3AHOME_GROUPING_SELECT_HOMES&superhost=false&title_type=SELECT_GROUPING&allow_override%5B%5D=&s_tag=ShdJZVjc'
    #SF
    #'https://www.airbnb.com/s/sanfran/plus_homes?refinement_paths%5B%5D=%2Fselect_homes&query=San%20Francisco%2C%20CA%2C%20United%20States&place_id=ChIJIQBpAG2ahYAR_6128GcTUEo&click_referer=t%3ASEE_ALL%7Csid%3A4e6ba24e-d983-43cb-b3d8-1d908609adef%7Cst%3AHOME_GROUPING_SELECT_HOMES&superhost=false&guests=0&adults=0&children=0&title_type=SELECT_GROUPING&allow_override%5B%5D=&s_tag=aD0c1Agz' 
    file = pd.read_csv('plus-links.csv')
    current_list = list(map(lambda x: x[1:-1], file['url'].values))

    def __init__(self):
        """
        Initialzing my driver for the selenium. This will open up a chrome browser
        specifically for automation. If you need to replicate, you need to make
        sure your chromedriver is in the parameter
        """
        #Mac
        self.driver = webdriver.Chrome('/Users/Work/Downloads/chromedriver')

        #Windows
        #self.driver = webdriver.Chrome('C:\\Users\\Paul\\Desktop\\chromedriver')


    def parse(self, response):
        self.driver.get(response.url)
        urls = []
        count = 348
        for y in range(0,17):
            try:
                r_time = random.randint(8,15)
                # Make sure page loads, this will fail and goes to except: 
                WebDriverWait(self.driver,r_time).until(EC.visibility_of_element_located((By.XPATH,'//a[@class="nav next taLnk ui_button primary"]')))
            except:
                # Looks for all of the links to the plus listings. This will not open them, but instead store them to urls for the next spider to use
                next = self.driver.find_elements_by_xpath('//a[@href]')
                print("START \n")
                for i in next:
                    if i.get_attribute("href")[:33] == "https://www.airbnb.com/rooms/plus":
                        if i.get_attribute("href") not in urls and i.get_attribute("href") not in self.current_list:
                            urls.append(i.get_attribute("href"))
                            air = AirbnbItem()
                            air['url'] = i.get_attribute("href")
                            air['url_num'] = count
                            count+=1
                            yield air

                try:
                    # Find the next page button and clicks it
                    next_page = self.driver.find_element_by_xpath('//li[@class="_r4n1gzb"]/a[@href]')
                    next_page.click()
                except:
                    # if fails, then ends the loop
                    print("THIS IS THE END")
                    print(y)
                    pass
            print(urls)

        print(urls)
