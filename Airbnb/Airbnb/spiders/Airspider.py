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


class AirspiderSpider(scrapy.Spider):
    name = 'Airspider'
    allowed_domains = ['airbnb.com']
    start_urls = ['https://www.airbnb.com/s/Los-Angeles--CA--United-States/plus_homes?refinement_paths%5B%5D=%2Fselect_homes&query=Los%20Angeles%2C%20CA%2C%20United%20States&place_id=ChIJE9on3F3HwoAR9AhGJW_fL-I&click_referer=t%3ASEE_ALL%7Csid%3Aecfaa229-a33b-44b7-a2ca-addf5ab3a639%7Cst%3AHOME_GROUPING_SELECT_HOMES&superhost=false&guests=0&adults=0&children=0&title_type=SELECT_GROUPING&allow_override%5B%5D=&s_tag=CWw7StG8'   ]
    #LA 
    """
    'https://www.airbnb.com/s/Los-Angeles--CA/plus_homes?refinement_paths%5B%5D=%2Fselect_homes&query=Los%20Angeles%2C%20CA&click_referer=t%3ASEE_ALL%7Csid%3Af95de7d2-8ffc-47e0-9aa3-a4de3dc3a722%7Cst%3AHOME_GROUPING_SELECT_HOMES&superhost=false&guests=0&adults=0&children=0&title_type=SELECT_GROUPING&allow_override%5B%5D=&s_tag=p3BdcKna'
    """
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
        count = 1
        for y in range(0,17):
            try:
                # Make sure page loads, this will fail and goes to except: 
                WebDriverWait(self.driver,10).until(EC.visibility_of_element_located((By.XPATH,'//a[@class="nav next taLnk ui_button primary"]')))
            except:
                # Looks for all of the links to the plus listings. This will not open them, but instead store them to urls for the next spider to use
                next = self.driver.find_elements_by_xpath('//a[@href]')
                print("START \n")
                for i in next:
                    if i.get_attribute("href")[:33] == "https://www.airbnb.com/rooms/plus":
                        if i.get_attribute("href") not in urls:
                            urls.append(i.get_attribute("href"))
                            air = AirbnbItem()
                            air['url'] = i.get_attribute("href")
                            air['url_num'] = count
                            count+=1
                            yield air

                try:
                    # Find the next page button and clicks it
                    next_page = self.driver.find_element_by_xpath('//li[@class="_b8vexar"]/a[@href]')
                    next_page.click()
                except:
                    # if fails, then ends the loop
                    print("THIS IS THE END")
                    print(y)
                    pass
        print(urls)
