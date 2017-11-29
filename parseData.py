#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from IPython.core.debugger import Tracer

from datetime import datetime

import numpy as np
import csv, json
import pandas as pd

import sys, traceback
reload(sys)
sys.setdefaultencoding('utf8')

################################################################################################
## Preparing DJIA data
# Reading DJIA index prices csv file
with open('../Stock_Market_Prediction/data/etherium_indices_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    # Converting the csv file reader to a lists 
    data_list = list(spamreader)

# Separating header from the data
header = data_list[0] 
data_list = data_list[1:] 

data_list = np.asarray(data_list)

# Selecting date and close value for each day
selected_data = data_list[:, [0, 3]]



df = pd.DataFrame(data=selected_data[0:,1:],
             index=selected_data[0:,0],
                                columns=[ 'adj close'],
                                        dtype='float64')
interpolated_df = df;
# # Reference for pandas interpolation http://pandas.pydata.org/pandas-docs/stable/missing_data.html
# # Adding missing dates to the dataframe
# df1 = df
# idx = pd.date_range('11-24-2016', '11-24-2017')
# df1.index = pd.DatetimeIndex(df1.index)
# df1 = df1.reindex(idx, fill_value=np.NaN)
# # df1.count() # gives 2518 count
# interpolated_df = df1.interpolate()
# interpolated_df.count() # gives 3651 count

# # Removing extra date rows added in data for calculating interpolation
# interpolated_df = interpolated_df[3:]

###############################################################################################  
## Preparing NYTimes data
# Function to parse and convert date format
date_format = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+%f"]
def try_parsing_date(text):
    for fmt in date_format:
        #return datetime.strptime(text, fmt)
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    raise ValueError('no valid date format found')

def addArticle(NYTimes_data, dict_keys, interpolated_df):
    articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
    articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
    #articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
    date = try_parsing_date(articles_dict['pub_date'])
    # print date
    current_article_str = articles_dict['headline']
    interpolated_df['articles'][date] += ". " + current_article_str.encode('utf8')
    # interpolated_df.set_value(date, 'articles', interpolated_df.loc[date, 'articles'] + '. ' + current_article_str)
    print date, ' ', len(interpolated_df['articles'][date])
    # For last condition in a year
    # if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1): 
    #     interpolated_df.set_value(date, 'articles', current_article_str)   




years = [2017,2016]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dict_keys = ['pub_date', 'headline'] #, 'lead_paragraph']
articles_dict = dict.fromkeys(dict_keys)
# Filtering list for type_of_material
type_of_material_list = ['blog', 'brief', 'news', 'editorial', 'op-ed', 'list','analysis']
# Filtering list for section_name
section_name_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health']
news_desk_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health', 'foreign']

current_date = datetime.now().strftime("%Y-%m-%d")
#years = [2015]
#months = [3]

current_article_str = ''      

## Adding article column to dataframe
interpolated_df["articles"] = ''
count_articles_filtered = 0
count_no_section_name = 0
count_no_news_desk = 0
count_no_material = 0
count_total_articles = 0
count_main_not_exist = 0               
count_unicode_error = 0     
count_attribute_error = 0   

for year in years:
    for month in months:
        if (year == 2017 and month == 12):
            continue
        # print year, " ", month
        file_str = '../Stock_Market_Prediction/data/my_nytimes_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str) as data_file:    
            NYTimes_data = json.load(data_file)
        count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
        # Parse a months worth of articles
        for i in range(len(NYTimes_data["response"]["docs"][:])):
            date = NYTimes_data["response"]["docs"][:][i]['pub_date']
            if (pd.to_datetime(date).date() < pd.to_datetime('2016-05-22').date()):
                continue
            if any('type_of_material' in NYTimes_data["response"]["docs"][:][i] and  NYTimes_data["response"]["docs"][:][i]['type_of_material'] != None and substring in NYTimes_data["response"]["docs"][:][i]['type_of_material'].lower() for substring in type_of_material_list):
                if any('section_name' in NYTimes_data["response"]["docs"][:][i] and NYTimes_data["response"]["docs"][:][i]['section_name'] != None and substring in NYTimes_data["response"]["docs"][:][i]['section_name'].lower() for substring in section_name_list):
                    # Do stuff here
                    addArticle(NYTimes_data, dict_keys, interpolated_df)
                    count_articles_filtered += 1

                else:
                    count_no_section_name += 1
                    # print 'not proper section name'
                    if('news_desk' in NYTimes_data["response"]["docs"][:][i]):
                        news_desk = 'news_desk'
                    else:
                        news_desk = 'new_desk'
                    if any(news_desk in NYTimes_data["response"]["docs"][:][i] and NYTimes_data["response"]["docs"][:][i][news_desk] != None and substring in NYTimes_data["response"]["docs"][:][i][news_desk].lower() for substring in news_desk_list):
                        #do same stuff here
                        addArticle(NYTimes_data, dict_keys, interpolated_df)
                        count_articles_filtered += 1
                    else:
                        count_no_news_desk += 1
                        # Tracer()()

            else:
                # print "Not type of material needed"
                count_no_material += 1
        Tracer()()
     
print count_articles_filtered 
print count_total_articles                     
print count_main_not_exist
print count_unicode_error



# ## Putting all articles if no section_name or news_desk not found
# for date, row in interpolated_df.T.iteritems():   
#     if len(interpolated_df.loc[date, 'articles']) <= 400:
#         #print interpolated_df.loc[date, 'articles']
#         #print date
#         date = datetime.strptime(date, '%Y-%m-%d').date()
#         month = date.month
#         year = date.year
#         file_str = '../Stock_Market_Prediction/data/my_nytimes_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
#         with open(file_str) as data_file:    
#             NYTimes_data = json.load(data_file)
#         count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
#         interpolated_df.set_value(date.strftime('%Y-%m-%d'), 'articles', '')
#         for i in range(len(NYTimes_data["response"]["docs"][:])):
#             try:
                
#                 articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
#                 articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
#                 #articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph       
#                 pub_date = try_parsing_date(articles_dict['pub_date'])
#                 #print 'article_dict: ' + articles_dict['headline']
#                 if date.strftime('%Y-%m-%d') == pub_date: 
#                     interpolated_df.set_value(pub_date, 'articles', interpolated_df.loc[pub_date, 'articles'] + '. ' + articles_dict['headline'])  
                
#             except KeyError:
#                 print 'key error'
#                 #print NYTimes_data["response"]["docs"][:][i]
#                 traceback.print_exc(file=sys.stdout)
#                 Tracer()()
#                 #count_main_not_exist += 1
#                 pass   
#             except TypeError:
#                 print "type error"
#                 #print NYTimes_data["response"]["docs"][:][i]
#                 traceback.print_exc(file=sys.stdout)
#                 Tracer()()
#                 #count_main_not_exist += 1
#                 pass


#>>> print count_articles_filtered 
#440770
#>>> print count_total_articles 
#1073132


## Filtering the whole data for a year
#filtered_data = interpolated_df.ix['2016-01-01':'2016-12-31']
#filtered_data.to_pickle('/Users/Dinesh/Documents/Project Stock predictions/data/pickled_ten_year_all.pkl')  


# Saving the data as pickle file
interpolated_df.to_pickle('../Stock_Market_Prediction/data/my_pickled_one_year_filtered_lead_para.pkl')  


# Save pandas frame in csv form
interpolated_df.to_csv('../Stock_Market_Prediction/data/my_sample_interpolated_df_one_years_filtered_lead_para.csv',
                       sep='\t', encoding='utf-8')



# Reading the data as pickle file
dataframe_read = pd.read_pickle('../Stock_Market_Prediction/data/my_pickled_one_year_filtered_lead_para.pkl')

print dataframe_read

# for i in range(0,len(NYTimes_data["response"]["docs"][:])):
    # if('section_name' in NYTimes_data["response"]["docs"][:][i]):
    #     print >> f, NYTimes_data["response"]["docs"][:][i]['pub_date'] + " " + NYTimes_data["response"]["docs"][:][i]['section_name'].encode('ascii', 'ignore')


#################################################################################

# Filtering rows
#filtered_data = interpolated_df.ix['2016-01-01':'2016-12-31']

# Syntax for accessing the data
#NYTimes_data["response"]["docs"][1:2][:]['headline']['main']
#NYTimes_data["response"]["docs"][1:2][0]['pub_date']
     

#    articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
#    try:
#        articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
#    except KeyError:
#        count_main_not_exist += 1
#        pass   
#    except TypeError:
#        count_main_not_exist += 1
#        pass


        
# Find out articles with less number of articles
# for date, row in interpolated_df.T.iteritems():   
#     if len(interpolated_df.loc[date, 'articles']) < 300:
#         print interpolated_df.loc[date, 'articles']
#         print date
           
            
 