import requests
import os
from itertools import combinations
from time import sleep
import cPickle as pickle
import json


# calls the googlemaps api to get the distances
def api_call(city1, city2):

    url = 'https://maps.googleapis.com/maps/api/distancematrix/' +\
          'json?units=imperial&origins=' +\
          '{}&destinations={}&key={}'.format(city1, city2, access_key)

    r = requests.get(url)
    dist = ''.join(r.json()['rows'][0]['elements'][0]['distance']
                   ['text'].split()[0].split(','))

    return float(dist)


# retrieves the distances, stores them in a dictionary, and save as a pickle
def get_distance():
    distance_dict = {}
    for combo in city_combos:
        distance_dict[combo] = api_call(combo[0], combo[1])
        sleep(5)

    with open('data/city_distances.pickle', 'w') as f:
        pickle.dump(distance_dict, f)

if __name__ == '__main__':
    # google api key
    access_key = os.environ['GOOGLE_KEY']

    # city_abbrv maps the NBA city abbreviations to the actual city names
    with open('data/city_abbrv.json', 'r') as f:
        city_abbrv = json.load(f)

    # create all possible pair combinations of the cities
    cities = city_abbrv.values()
    city_combos = combinations(cities, r=2)

    get_distance()