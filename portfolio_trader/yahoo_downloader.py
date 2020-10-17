from portfolio_trader import *


class YahooFinanceHistory:
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, date_from, date_to, timeout):
        self.symbol = symbol
        self.session = requests.Session()
        self.date_from = date_from
        self.date_to = date_to
        self.timeout = timeout
        self.crumb = None

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        # set the starting and end dates
        date_to = int(time.mktime(pd.Timestamp(self.date_to).timetuple()))
        date_from = int(time.mktime(pd.Timestamp(self.date_from).timetuple()))
        url = self.quote_link.format(quote=self.symbol, dfrom=date_from,
                                     dto=date_to, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])

