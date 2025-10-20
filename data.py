import wrds
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1️⃣ Connect to WRDS
# --------------------------------------------------
db = wrds.Connection()


#opprcd2023
#secprd2023
#optionmnames
#zero_curve


# Extract only ticker and secid
#df = db.raw_sql("""
#    SELECT DISTINCT ticker, secid
#    FROM optionm.optionmnames
#""")

# Save to CSV
#df.to_csv('optionmnames_ticker_secid.csv', index=False)
#print("Data saved to optionmnames_ticker_secid.csv")


## First, let's see what columns are available in secprd2023
#print("Columns in secprd2023 table:")
#columns_info = db.raw_sql("""
#    SELECT column_name, data_type 
#    FROM information_schema.columns 
#    WHERE table_schema = 'optionm' 
#    AND table_name = 'secprd2023'
#    ORDER BY ordinal_position
#""")
#print(columns_info)





aapl_id = 101594  # replace with your actual secid

## Get the underlying daily prices
#df_underlying = db.raw_sql(f"""
#    SELECT date, close
#    FROM optionm.secprd2023
#    WHERE secid = {aapl_id}
#    ORDER BY date
#""")
#
##save to csv
#df_underlying.to_csv('aapl_underlying_prices.csv', index=False)

#df_options_columns = db.raw_sql("""
#    SELECT column_name, data_type 
#    FROM information_schema.columns 
#    WHERE table_schema = 'optionm' 
#    AND table_name = 'opprcd2023'
#    ORDER BY ordinal_position
#""")
#
#print("Columns in opprcd2023 table:")
#print(df_options_columns)

df_options = db.raw_sql(f"""
    SELECT date, exdate, cp_flag, strike_price,
           best_bid, best_offer, impl_volatility
    FROM optionm.opprcd2023
    WHERE secid = {aapl_id}
    AND date BETWEEN '2023-01-01' AND '2023-01-31'
""")

print(df_options.head())



import pandas as pd

# Convert strike price to actual (divide if scaled)
if df_options['strike_price'].max() > 10000:  # detect scaling
    df_options['strike_price'] /= 1000

# Compute mid prices
df_options['mid'] = (df_options['best_bid'] + df_options['best_offer']) / 2

# Compute time to maturity in years
df_options['T'] = (pd.to_datetime(df_options['exdate']) - pd.to_datetime(df_options['date'])).dt.days / 365

# Filter out invalid or expired options
df_options = df_options[(df_options['T'] > 0) & (df_options['mid'] > 0)]


df_underlying = db.raw_sql(f"""
    SELECT date, close
    FROM optionm.secprd2023
    WHERE secid = {aapl_id}
""")



df_merged = pd.merge(df_options, df_underlying, on='date', how='inner')
df_merged.rename(columns={'close': 'S'}, inplace=True)

print(df_merged.head())

df_merged.to_csv('aapl_options_jan2023.csv', index=False)

