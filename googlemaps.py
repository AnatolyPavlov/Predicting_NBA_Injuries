import requests
import os
from itertools import combinations
from time import sleep


def get_distance(city1, city2):

    url = 'https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins={}&destinations={}&key={}'.format(city1, city2, access_key)
    r = requests.get(url)
    dist = ''.join(r.json()['rows'][0]['elements'][0]['distance']\
            ['text'].split()[0].split(','))
    return float(dist)

access_key = os.environ['GOOGLE_KEY']

city_abbrv = {'ATL': 'Atlanta',
              'BOS': 'Boston',
              'CLE': 'Cleveland',
              'GSW': 'Oakland',
              'OKC': 'Oklahoma+City',
              'SAS': 'San+Antonio',
              'POR': 'Portland',
              'MIA': 'Miami',
              'IND': 'Indiana',
              'TOR': 'Toronoto',
              'CHI': 'Chicago',
              'BKN': 'Brooklyn',
              'CHA': 'Charlotte',
              'NYK': 'New+York',
              'DET': 'Detroit',
              'PHI': 'Philadelphia',
              'ORL': 'Orlando',
              'MIL': 'Milwaukee',
              'WAS': 'Washington+DC',
              'DEN': 'Denver',
              'DAL': 'Dallas',
              'MIN': 'Minnesota',
              'LAC': 'Los+Angeles',
              'HOU': 'Houston',
              'LAL': 'Los+Angeles',
              'MEM': 'Memphis',
              'PHX': 'Phoenix',
              'NOP': 'New+Orleans',
              'UTA': 'Utah',
              'SAC': 'Sacramento'}

cities = city_abbrv.values()
city_combos = combinations(cities, r=2)
distance_dict = {}
i = 0
for combo in city_combos:
    distance_dict[combo] = get_distance(combo[0], combo[1])
    sleep(5)
    print i / 900.
    i += 1