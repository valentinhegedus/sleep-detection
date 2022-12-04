from datetime import datetime


class Logger:

    def __init__(self, verbose):
        self._verbose = verbose

    def log(self, event, data):
        if self._verbose is False:
            return
        date = datetime.today().strftime('%Y/%m/%d %H:%M:%S')
        print(f"[{date}]\t[{event}]\t{data}")
