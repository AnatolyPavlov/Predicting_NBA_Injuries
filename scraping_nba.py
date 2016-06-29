import requests
from time import sleep
import json
import random
import cPickle as pickle


# Request to get the URL fails often so using while loop to check
# for bad request and also choose random headers to trick nba.com
def get_response(url):

    user_agent = ['Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
                  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.3319.102 Safari/537.36',
                'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; PPC Mac OS X x.y; rv:10.0) Gecko/20100101 Firefox/10.0',
                'Mozilla/5.0 (Maemo; Linux armv7l; rv:10.0) Gecko/20100101 Firefox/10.0 Fennec/10.0',
                'Mozilla/5.0 (Windows NT x.y; rv:10.0) Gecko/20100101 Firefox/10.0',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.517 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/4E423F',
                'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36']

    headers = {'User-Agent': random.choice(user_agent)}

    status = 'Bad Request'
    while status == 'Bad Request':
        r = requests.get(url, headers=headers)
        status = r.reason

        # sleep for 10 secs before trying again
        sleep(random.choice([1, 2, 3, 5, 10, 12]))
        print 'Still scraping....'
    return r


# show progress of scraping
def print_progress(season):
    print 'Finished scraping {}'.format(season)


# scraping players' drives and speeds
def drives_speed():

    seasons = range(2013, 2016)
    measures = ['Drives', 'SpeedDistance']

    for season in seasons:
        for measure in measures:

            url = 'http://stats.nba.com/stats/leaguedashptstats?College=' +\
                  '&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=' +\
                  '&DraftYear=&GameScope=&Height=&LastNGames=0&LeagueID=' +\
                  '00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=' +\
                  '0&PerMode=PerGame&PlayerExperience=&PlayerOrTeam=' +\
                  'Player&PlayerPosition=&PtMeasureType=' +\
                  measure +\
                  '&Season=' +\
                  str(season) + '-' + str(season+1)[2:] +\
                  '&SeasonSegment=&SeasonType=Regular+Season&StarterBench=' +\
                  '&TeamID=0&VsConference=&VsDivision=&Weight='

            r = get_response(url)

            # saving json
            with open('data/{}_{}.json'.format(measure, season), 'w') as f:
                json.dump(r.json(), f)

            print_progress(season)

    return 'All scraped!'

# scraping gamelogs starting from 1990
def gamelogs():

    seasons = range(1990, 2016)

    for season in seasons:
        url = 'http://stats.nba.com/stats/leaguegamelog?Counter=' +\
              '1000&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=' +\
              str(season) + '-' + str(season+1)[2:] +\
              '&SeasonType=Regular+Season&Sorter=PTS'

        r = get_response(url)

        # saving json
        with open('data/{}_gamelog.json'.format(season), 'w') as f:
            json.dump(r.json(), f)

        print_progress(season)

    return 'All scraped!'


# scraping players' heights and weights since the 1990
def heights_weights():

    seasons = range(1990, 2016)

    for season in seasons:
        url = 'http://stats.nba.com/stats/leaguedashplayerbiostats?College=' +\
              '&Conference=&Country=&DateFrom=&DateTo=&Division=' +\
              '&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=' +\
              '&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID' +\
              '=0&Outcome=&PORound=0&PerMode=PerGame&Period=' +\
              '0&PlayerExperience=&PlayerPosition=&Season=' +\
              str(season) + '-' + str(season+1)[2:] +\
              '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=' +\
              '&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='

        r = get_response(url)

        # saving json
        with open('data/{}_heights_weights.json'.format(season), 'w') as f:
            json.dump(r.json(), f)

        print_progress(season)

    return 'All scraped!'


# scraping players' heights and weights since the 1990
def season_stats():

    seasons = range(1990, 2016)

    for season in seasons:
        url = 'http://stats.nba.com/stats/leaguedashplayerstats?College=' +\
        '&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=' +\
        '&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=' +\
        '0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=' +\
        '0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' +\
        '0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=' +\
        str(season) + '-' + str(season+1)[2:] +\
        '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=' +\
        '&TeamID=0&VsConference=&VsDivision=&Weight='

        r = get_response(url)

        # saving json
        with open('data/{}_season_stats.json'.format(season), 'w') as f:
            json.dump(r.json(), f)

        print_progress(season)

    return 'All scraped!'

# scraping game_ids since 2013
def get_game_ids():

    seasons = range(2013, 2016)

    game_ids = []
    for season in seasons:
        url = 'http://stats.nba.com/stats/leaguegamelog?Counter=' +\
        '1000&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=' +\
        str(season) + '-' + str(season+1)[2:] +\
        '&SeasonType=Regular+Season&Sorter=PTS'

        r = get_response(url)

        data = r.json()
        rows = data['resultSets'][0]['rowSet']

        game_ids.append(set([row[5] for row in rows]))

        print_progress(season)

    # saving as pickle file
    with open('data/game_ids.pickle', 'w') as f:
        pickle.dump(game_ids, f)

    return 'All scraped!'


# scraping pace stats since 2013
def pace_stats():

    with open('data/game_ids.pickle', 'r') as f:
        game_ids = pickle.load(f)
    seasons = range(2013, 2016)

    i = 0
    for season in seasons:
        games_scraped = 1
        json_list = []
        for game_id in game_ids[i]:
            url = 'http://stats.nba.com/stats/boxscoreadvancedv2?EndPeriod=' +\
            '10&EndRange=28800&GameID=' +\
            str(game_id) +\
            '&RangeType=0&Season=' +\
            str(season) + '-' + str(season+1)[2:] +\
            '&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'

            r = get_response(url)
            json_list.append(r.json())

            print season, games_scraped
#            print_progress(season)
            games_scraped += 1

        # saving as pickle
        with open('data/{}_pace_stats.pickle'.format(season), 'w') as f:
            pickle.dump(json_list, f)
        i += 1

    return 'All scraped!'

# scraping tracking stats since 2013
def tracking_stats():

    with open('data/game_ids.pickle', 'r') as f:
        game_ids = pickle.load(f)
    seasons = range(2013, 2016)

    i = 0
    for season in seasons:
        games_scraped = 1
        json_list = []
        for game_id in game_ids[i]:
            url = 'http://stats.nba.com/stats/boxscoreadvancedv2?EndPeriod=' +\
            '10&EndRange=28800&GameID=' +\
            str(game_id) +\
            '&RangeType=0&Season=' +\
            str(season) + '-' + str(season+1)[2:] +\
            '&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'

            r = get_response(url)
            json_list.append(r.json())

            print season, games_scraped
#            print_progress(season)

            # saving as pickle
        with open('data/{}_tracking_stats.pickle'.format(season), 'w') as f:
            pickle.dump(json_list, f)

        i += 1

    return 'All scraped!'

if __name__ == '__main__':

    drives_speed()
    gamelogs()
    heights_weights()
    season_stats()
    get_game_ids()
    pace_stats()
    tracking_stats()