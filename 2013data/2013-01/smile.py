from pathlib import Path
import pandas as pd

csv_path = Path(__file__).parent / '2013-01-02options.csv'
print('Loading:', csv_path)
df = pd.read_csv(csv_path)
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt

# Example filter: underlying="AAPL", option type call
df_call = df[(df['underlying'] == 'MSFT') & (df['type'] == 'call')]

df_call = df_call.sort_values('strike')



plt.figure(figsize=(8,5))
plt.plot(df_call['strike'], df_call['implied_volatility'], marker='o', linestyle='-')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility Smile for MSFT Calls')
plt.grid(True)
plt.show()

