import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplfinance as mpf

# ohlc: DataFrame indexed by datetime with columns Open,High,Low,Close,Volume
style = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(up="tab:green", down="tab:red", edge="inherit", wick="inherit"),
    gridstyle="-", gridcolor="0.85"
)





