import pandas as pd
import requests
from bs4 import BeautifulSoup


# scraping for injuries
def get_injuries():

    # Create columns for dataframe
    df = pd.DataFrame(columns=['Date', 'Team', 'Player', 'Notes'])

    # url for Prosportstransactions
    url = 'http://www.prosportstransactions.com/basketball/Search/' +\
          'SearchResults.php?Player=&Team=&BeginDate=&EndDate=' +\
          '&InjuriesChkBx=yes&Submit=Search&start='

    # loop through all available pages and store info in a dataframe
    for i in xrange(0, 24600, 25):
        r = requests.get(url+str(i))
        soup = BeautifulSoup(r.content)

        # Read from html
        transactions = [line.text.encode('ascii', 'ignore')
                        for line in soup.findAll('td')]
        # Remove irrelevant content
        del transactions[transactions.index(''):]

        # Populate dataframe
        for i in xrange(5, len(transactions)-4, 5):
            df = df.append({'Date': transactions[i],
                            'Team': transactions[i+1],
                            'Player': transactions[i+3],
                            'Notes': transactions[i+4]},
                            ignore_index=True)

    # save to csv
    df.to_csv('../data/injuries.csv')

if __name__ == '__main__':
    get_injuries()