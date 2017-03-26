import logging
import pandas as pd


log_fmt = "\n%(asctime)s\n======================\n%(message)s\n======================\n"
logging.basicConfig(format=log_fmt, level=logging.DEBUG)

pd.options.mode.chained_assignment = None
