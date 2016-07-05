import requests
from time import sleep
import json
import random
import cPickle as pickle


class NBA_Scraper(object):

    def __init__(self, start_season, end_season, show_progress=False):
        self.start_season = start_season
        self.end_season = end_season
        self.seasons = range(self.start_season, self.end_season)
        self.show_progress = show_progress

    # gets response from a URL
    def get_response(self, url):
        # URL request fails often so choose random headers for each request
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

        # Keep on trying if a bad request is received
        status = 'Bad Request'
        while status == 'Bad Request':
            r = requests.get(url, headers=headers)
            status = r.reason

            # Selecting random sleep time
            sleep(random.choice([1, 2, 3, 5, 10, 12]))
            print 'Still scraping....'
        return r

    # prints the progress of the scraping
    def print_progress(self, season):
        print 'Finished scraping {}'.format(season)

    # scrapes for the gamelogs
    def get_gamelogs(self):

        # iterates through the specified seasons
        for season in self.seasons:
            url = 'http://stats.nba.com/stats/leaguegamelog?Counter=' +\
                  '1000&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=' +\
                  str(season) + '-' + str(season+1)[2:] +\
                  '&SeasonType=Regular+Season&Sorter=PTS'

            r = self.get_response(url)

            # saves json
            with open('../data/{}_gamelog.json'.format(season), 'w') as f:
                json.dump(r.json(), f)

            if self.show_progress:
                self.print_progress(season)

        return 'All scraped!'

    # scrapes heights and weights
    def get_heights_weights(self):

        # iterates through the specified seasons
        for season in self.seasons:
            url = 'http://stats.nba.com/stats/leaguedashplayerbiostats?College=' +\
                  '&Conference=&Country=&DateFrom=&DateTo=&Division=' +\
                  '&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=' +\
                  '&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID' +\
                  '=0&Outcome=&PORound=0&PerMode=PerGame&Period=' +\
                  '0&PlayerExperience=&PlayerPosition=&Season=' +\
                  str(season) + '-' + str(season+1)[2:] +\
                  '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=' +\
                  '&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='

            r = self.get_response(url)

            # saves json
            with open('../data/{}_heights_weights.json'.format(season), 'w') as f:
                json.dump(r.json(), f)

            if self.show_progress:
                self.print_progress(season)

        return 'All scraped!'

    # scrapes season stats
    def get_season_stats(self):

        # iterates through the specified seasons
        for season in self.seasons:
            url = 'http://stats.nba.com/stats/leaguedashplayerstats?College=' +\
            '&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=' +\
            '&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=' +\
            '0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=' +\
            '0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=' +\
            '0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=' +\
            str(season) + '-' + str(season+1)[2:] +\
            '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=' +\
            '&TeamID=0&VsConference=&VsDivision=&Weight='

            r = self.get_response(url)

            # saves json
            with open('../data/{}_season_stats.json'.format(season), 'w') as f:
                json.dump(r.json(), f)

            if self.show_progress:
                self.print_progress(season)

        return 'Season stats all scraped!'

    # scrapes game_ids
    def get_game_ids(self):

        self.game_ids = []

        # iterates through the specified seasons
        for season in self.seasons:
            url = 'http://stats.nba.com/stats/leaguegamelog?Counter=' +\
            '1000&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=' +\
            str(season) + '-' + str(season+1)[2:] +\
            '&SeasonType=Regular+Season&Sorter=PTS'

            r = self.get_response(url)

            data = r.json()
            rows = data['resultSets'][0]['rowSet']

            # saves the unique game ids to the a list
            self.game_ids.append(set([row[5] for row in rows]))

            if self.show_progress:
                self.print_progress(season)
        if self.show_progress:
            print "Game ID's all scraped!"

    # scrapes pace or tracking stats
    def get_pace_tracking(self, stat):

        i = 0
        game_ids = self.get_game_ids()

        # iterates through the game id's in the specified seasons
        for season in self.seasons:
            games_scraped = 1
            json_list = []
            for game_id in game_ids[i]:

                if stat == 'pace':
                    url = 'http://stats.nba.com/stats/boxscoreadvancedv2?' +\
                    'EndPeriod=10&EndRange=28800&GameID=' +\
                    str(game_id) +\
                    '&RangeType=0&Season=' +\
                    str(season) + '-' + str(season+1)[2:] +\
                    '&SeasonType=Regular+Season&StartPeriod=1&StartRange=0'
                elif stat == 'tracking':
                    url = 'http://stats.nba.com/stats/boxscoreplayertrackv2?EndPeriod=' +\
                    '10&EndRange=55800&GameID=' +\
                    str(game_id) +\
                    '&RangeType=2&Season=' +\
                    str(season) + '-' + str(season+1)[2:] +\
                    '&SeasonType=Playoffs&StartPeriod=1&StartRange=0'
                else:
                    return "Stat must be either be 'pace' or 'tracking'"

                # saves the jsons in a list
                r = self.get_response(url)
                json_list.append(r.json())

                if self.show_progress:
                    print season, games_scraped
                games_scraped += 1

            i += 1

            # saves as a pickle
            with open('../data/{}_{}_stats.pickle'.format(season, stat), 'w') as f:
                pickle.dump(json_list, f)
