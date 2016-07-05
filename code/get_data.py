import scrape_nba
import scrape_pst
import scrape_googlemaps

def get_data():
    nba = scrape_nba.NBA_Scraper(2013, 2016)
    nba.get_gamelogs()
    nba.get_heights_weights()
    nba.get_seasons_stats()
    nba.get_pace_tracking('pace')
    nba.get_pace_tracking('tracking')
    scrape_pst.get_injuries()
    scrape_googlemaps.get_distance()

if __name__ == '__main__':
    get_data()