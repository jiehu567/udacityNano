
# coding: utf-8

# ## OpenStreetMap Project for Nano degree
# *by Jie Hu, jie.hu.ds@gmail.com*
# 
# This is a project for Udacity [Nano-degree](https://classroom.udacity.com/nanodegrees/) P3 part. 
# In this project I will clean the open street data of [San Francisco Bay Area](https://mapzen.com/data/metro-extracts/metro/san-francisco-bay_california/) and write into MongoDB database, after which yields some visualization work. 
# 
# The structure of this report is:
# 
# 1. Problems in dataset
# 2. Data Cleaning
# 3. Data Overview and Wrangling with MongoDB
# 4. Additional Ideas
# 5. Conclusion
# 
# 
# ### Part 1. Problems in dataset
# 

# In[1]:

# import package
# Load packages and raw data file
import re
from collections import defaultdict
import json
import xml.etree.cElementTree as ET
import pprint
import numpy as np
import datetime as dt

osm_file = 'sf_sample.osm'


# After I get sample data, I find out there are several problems in the dataset.
# 
# 1. data are listed within different tags, which might have different structure: 

# In[2]:

# type and number of nodes
dic = defaultdict(int)

for _, elem in ET.iterparse(osm_file):
    dic[elem.tag] +=1
print dic


# 2. keys, like 'exit_to', 'exit_to:left', 'exit_to:right' can be involved in one list
# 3. keys, like 'lat' and 'lon' can be saved in a dictionary to seperate from other attributes:
# ~~~~
# {
#     'position': {'lat': ..., 
#                  'lon': ...
#                  }
# }
# ~~~~
# 4. value of 'timestamp' is just a string, which can be transformed into: 
# ~~~~
# {
#     'time': {'year':...,
#              'month':...,
#              'day':...,
#              ...}
# }
# ~~~~
# 5. keys, like 'sfgov.org', 'addr.source' can not be used in MongoDB because they include '.'
# 6. keys, like 'addr:country', 'addr:state' should be transformed into:
# 
# ~~~~
# {
#     'addr': {'city': ...
#              'country':...
#              'state':...
#  ...
# }
# ~~~~
# instead of:
# ~~~~
# {
#     'addr:city':...
#     'addr:country':...
#     'addr:state':...
# }
# ~~~~
# 
# 
# ### Part 2. Data Cleaning
# 
# Then I check data structure of each node, and write a function to process each part respectively.
# 
# **Problems 1-4:**
# 
# The method 'attach_node' will process node according to its type and problems and then save into a dictionary, 'temp_dic'. Then after this, the processed data of such node, in the form of dictionary will be appended to a list.

# In[3]:

def attach_node(temp_dic, node_name):
    for node in elem.iter(node_name):
            # add uid, version, changeset, user, id
            for key in ['uid', 'version', 'changeset', 'user', 'id']:    
                if node.attrib.has_key(key):
                    temp_dic[key] = node.attrib[key]

            # add position with latitude and longitude
            if node.attrib.has_key('lat') and node.attrib.has_key('lon'):
                temp_dic['position'] = {'lat': float(node.attrib['lat']),
                                        'lon': float(node.attrib['lon'])}

            # timestamp into year, month, day, hour, minute, second
            if node.attrib.has_key('timestamp'):
                time = dt.datetime.strptime(node.attrib['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
                temp_dic['time'] = {'year': time.year,
                                    'month': time.month,
                                    'day': time.day,
                                    'hour': time.hour,
                                    'minute': time.minute}         # Here I think not necessary to extract second


            # attach data from key 'tag'
            for tag in node.iter('tag'):

                # attach tag value IN exit_to list
                exit_arr = []
                for key in ['exit_to','exit_to:left','exit_to:right']:
                    if tag.attrib['k'] == key:
                        exit_arr.append(tag.attrib['v'])

                if exit_arr != []:
                    temp_dic['exit_to'] = exit_arr

                # attach tag value NOT IN exit_to list
                if not tag.attrib['k'] in ['exit_to','exit_to:left','exit_to:right']:
                    temp_dic[tag.attrib['k']] = tag.attrib['v']

            temp_dic['node_type'] = node_name


# In[4]:

# Process data

dic_sample = []

for _, elem in ET.iterparse(osm_file):

    temp_dic = {}
    
    
    # attach data from key 'member'
    for member in elem.iter('member'):
        for key in member.keys():
            if member.attrib[key] != "":
                temp_dic[key] = member.attrib[key]
            else:
                temp_dic[key] = None
    
    # attach data from key 'node'
    attach_node(temp_dic, 'node')
    
    # attach data from key 'relation', it's exactly the same as 'node'
    attach_node(temp_dic, 'relation')
    
    # attach data from key 'way', it's exactly the same as 'node'
    attach_node(temp_dic, 'way')
    
    if temp_dic != {}:
        dic_sample.append(temp_dic)
        


# Now dic_sample almost have cleaned data:

# In[7]:

dic_sample[:5]


# ** Problem 5**
# - keys, like 'sfgov.org', 'addr.source' can not be used in MongoDB because they include '.'
# - keys, like 'addr:country', 'addr:state' should be transformed into:
# 
# ~~~~
# {
#     'addr': {'city': ...
#              'country':...
#              'state':...
#  ...
# }
# ~~~~
# 
# Then will be solved by below replacement, my idea is not just process address data, but also all data structured in such way, now I need to check what these key like:

# In[8]:

post_style = re.compile(r'.*:.*')
key_set = set()
for item in dic_sample:
    for key, value in item.items():
        if post_style.match(key) and key not in key_set:
            key_set.add(key)

# print key_set


# Result of print:
# ~~~~
# {'FG:COND_INDEX', 
#  'FG:GPS_DATE',
#  ...
#  'Tiger:HYDROID',
#  'Tiger:MTFCC',
#  'abandoned:place',
#  ...
#  'addr:city',
#  ..
#  }
# ~~~~
# 
# Here's the method to do the processing:

# In[19]:

# change key:sub_key pairs into key: {subkey1:..., subkey2:...,...}

re_style = re.compile(r'.+:.+')

for item in dic_sample:
    
    key_set = set()
    
    for item_key in item.keys():
        if re_style.match(item_key):
            item_subkeys = item_key.split(':')
            if len(item_subkeys) >= 2:
                if item_subkeys[0] not in key_set:
                    key_set.add(item_subkeys[0])
                    item[item_subkeys[0]] = {}
            
                item[item_subkeys[0]][item_subkeys[1]] = item.pop(item_key)
                    
            


# ** Problem 6**
# 
# Some keys, for example: {'addr.source', 'sfgov.org'} is not suitable to be used in MongoDB so I replace them.

# In[20]:

post_style = re.compile(r'.+\..+')
for item in dic_sample:
    for key, value in item.items():
        if key == 'addr.source':
            if item.has_key('addr'):
                item['addr']['source'] = item.pop('addr.source')
            else:
                item['addr_source'] = item.pop('addr.source')
        elif key == 'sfgov.org':
            item['sfgov'] = item.pop('sfgov.org')


# Now dic_sample saved our data in a pretty good format, so I can store it into a JSON file.

# In[18]:

import json
with open('data_example.json', 'w') as fp:
    json.dump(dic_sample, fp)


# ### Part 3. Data Overview and Wrangling with MongoDB
# 
# 
# Now start my MongoDB, connect with python client, and insert the data
# 

# In[21]:

from pymongo import MongoClient


# In[25]:

client = MongoClient('localhost:27017')
db = client.openstreetmap

for item in dic_sample:
    db.openstreetmap.insert_one(item)
    


# In[24]:

# db.openstreetmap.drop()


# Basic data information:
# 
# sf_sample.osm ..................... 102.5 MB
# san-francisco_california.osm........1.01  GB    
# 
# (Because it takes too long time to process whole data on my old mac, here I will only use sample data, which is large enough for this project. Ideas will be the same)
# 
# data_example.json .... 132.1 MB
#                                                 
# #### Number of documents

# In[26]:

db.openstreetmap.find().count()


# #### Number of nodes

# In[34]:

db.openstreetmap.find({"node_type":"node"}).count()


# #### Number of way

# In[38]:

db.openstreetmap.find({"node_type":"way"}).count()


# All are exactly match what we get in front of this report:
# 
# {'node': 471488, ..., 'way': 55115})
# 
# Number of 'way' is almost same, I will just ignore the only one difference of count.
# 
# #### Number of unique users

# In[48]:

len(db.openstreetmap.distinct('user'))


# Here I use below methods to write my query pipelines:

# In[50]:

def aggregate(db, pipeline):
    return [doc for doc in db.openstreetmap.aggregate(pipeline)]


# In[51]:

def make_pipeline():
    pipeline = [{"$group": {"_id": "$user",
                                        "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}},
                            {"$limit": 10}
]
    return pipeline


# #### Top 10 most contribution users

# In[52]:

import pprint
pipeline = make_pipeline()
result = aggregate(db, pipeline)
pprint.pprint(result)


# ### Part 4: Additional Ideas

# #### Most contribution hour

# In[53]:

def make_pipeline():
    pipeline = [{"$group": {"_id": "$time.hour",
                                        "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}},
                            {"$limit": 24}
]
    return pipeline

pipeline = make_pipeline()
result = aggregate(db, pipeline)
result_dic = {}
for item in result:
    result_dic[item['_id']] = item['count']


# In[71]:

import matplotlib.pyplot as plt

values = result_dic.values()
clrs = ['#01cab4' if (x < max(values)) else '#ff3f49' for x in values ]

plt.bar(range(len(result_dic)), values, align='center', color = clrs)
plt.title("Most Contribution Hour")
plt.xlabel("Hour")
plt.ylabel("Contribution")
plt.xlim(-1,24)
plt.show()


# #### most contribution month

# In[73]:

def make_pipeline():
    pipeline = [{"$group": {"_id": "$time.month",
                                        "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}},
                            {"$limit": 24}
]
    return pipeline

pipeline = make_pipeline()
result = aggregate(db, pipeline)
result_dic = {}
for item in result:
    result_dic[item['_id']] = item['count']

values = result_dic.values()
clrs = ['#01cab4' if (x < max(values)) else '#ff3f49' for x in values ]

plt.bar(range(len(result_dic)), values, align='center', color = clrs)
plt.title("Most Contribution Month")
plt.xlabel("Month")
plt.ylabel("Contribution")
plt.xlim(-1,13)
plt.show()


# #### Top 10 appearing amenities

# In[77]:

def make_pipeline():
    pipeline = [{"$match":{"amenity":{"$exists":1}}},
                {"$group": {"_id": "$amenity",
                                        "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
]
    return pipeline


pipeline = make_pipeline()
result = aggregate(db, pipeline)
result


# #### Most popular cuisines

# In[80]:

def make_pipeline():
    pipeline = [{"$match":{"amenity":{"$exists":1}}},
                {"$group": {"_id": "$cuisine",
                            "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
]
    return pipeline


pipeline = make_pipeline()
result = aggregate(db, pipeline)
result


# While ignoring the None type, we can say that the coffee shops are most popular cuisine.

# #### Part 4: Conclusion
# 

# This project offers me very effecient way to wrangle data. I've learned extracting, transforming and loading technicals and I found a lot of interesting ideas in the city I'm living. I'm amazed by the data produced by people.
# 
# Regarding the possible improvements of this report, I think below approaches might be a good choice:
# - Time distribution of top10 contributors
# - Most frequent contribution regions, I think bounding boxes of district in San Francisco are required
# - Distribution of most popular cuisines, we can see this from how people with different culture background might be distributed

# In[ ]:



