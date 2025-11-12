import os
import pandas as pd
import re
folder_path = r'2013-01'


#here

output_options_file = 'A_optiondata_2013-01.csv'
if os.path.exists(output_options_file):
    print(f"{output_options_file} already exists — skipping options processing.")
else:
    A_optiondata = []
    for filename in os.listdir(folder_path):
        if filename.endswith("options.csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)  # read CSV
            # ...existing code...
            # safe filter + avoid SettingWithCopyWarning
            df_A = df[(df['underlying'] == 'A') & (df['type'].str.lower() == 'call')].copy()
            # ensure bid/ask exist and are numeric
            for col in ('bid', 'ask'):
                if col in df_A.columns:
                    df_A[col] = pd.to_numeric(df_A[col], errors='coerce')
                else:
                    df_A[col] = pd.NA
            # compute midpoint
            df_A['mid'] = (df_A['bid'] + df_A['ask']) / 2
            df_A['expiration'] = pd.to_datetime(df_A['expiration'], errors='coerce')
            df_A['quote_date'] = pd.to_datetime(df_A['quote_date'], errors='coerce')

            # Time to maturity in years
            df_A['T'] = (df_A['expiration'] - df_A['quote_date']).dt.days / 365.0
            df_A['r'] = 0.1

            # ...existing code... # filter underlying 'A'
            if not df_A.empty:
                A_optiondata.append(df_A)  # append filtered DF
            
            # combine all filtered DataFrames
            if A_optiondata:
                A_optiondata_df = pd.concat(A_optiondata, ignore_index=True)
                # save to CSV
                A_optiondata_df.to_csv('A_optiondata_2013-01.csv', index=False)
                print("Saved filtered data to A_optiondata_2013-01.csv")
            else:
                print("No option data found for underlying 'A'")




output_spot_file = 'A_spot_2013-01.csv'
if os.path.exists(output_spot_file):
    print(f"{output_spot_file} already exists — skipping spot processing.")
else:
    A_spot = []
    for filename in os.listdir(folder_path):
        if filename.endswith("stocks.csv"):

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)  # read CSV
            # add filename (without extension) as 'date' column
            base = os.path.splitext(filename)[0]
            m = re.search(r'\d{4}-\d{2}-\d{2}', base)
            df['quote_date'] = m.group(0) if m else base
            # select symbol, close and date, rename for consistency
            df_S = df[['symbol', 'close', 'quote_date']].rename(columns={'symbol': 'underlying'})
            df_S = df_S[df_S['underlying'] == 'A'].copy()
            if not df_S.empty:
                A_spot.append(df_S)

    # combine all filtered DataFrames
    if A_spot:
        A_spot_df = pd.concat(A_spot, ignore_index=True)
        # save to CSV
        A_spot_df.to_csv(output_spot_file, index=False)
        print("Saved filtered data to A_spot_2013-01.csv")
    else:
        print("No spot data found for underlying 'A'")

merged_file = 'A_df.csv'
if os.path.exists(merged_file):
    print(f"{merged_file} already exists — skipping spot processing.")
else:
    A_optiondata_df = pd.read_csv('A_optiondata_2013-01.csv')
    A_spot_df = pd.read_csv('A_spot_2013-01.csv')

    merge_data = pd.merge(A_optiondata_df, A_spot_df, on=['underlying', 'quote_date'], how='left')

    merge_data.to_csv('A_df.csv')

    print(merge_data.head())


#exp volatility as sigma element of [0,infinity)


import math
import numpy as np
from scipy.stats import norm

df = pd.read_csv('A_df.csv')

# normalize column names: 'close' -> 'spot', 'mid' -> 'mid_price'
if 'close' in df.columns and 'spot' not in df.columns:
    df = df.rename(columns={'close': 'spot'})
if 'mid' in df.columns and 'mid_price' not in df.columns:
    df = df.rename(columns={'mid': 'mid_price'})

# required columns
required = ['spot', 'strike', 'T', 'r', 'mid_price']
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# select columns and add weight column
df = df[required].copy()
df['w'] = 1.0


rows = df.to_dict(orient='records')

def bs_call_price(S,K,T,r,sigma):
    from scipy.stats import norm
    if T <= 0:
        return max(S-K,0)
    d1 = (np.log(S/K) +(r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) -K*np.exp(-r*T)*norm.cdf(d2)

print(bs_call_price(100,100,0.1,0.1,0.2))