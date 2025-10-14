import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Commodity:
    def __init__(self):
        return None
    
    def _read_csv(self):
        """Reads a CSV file and returns a DataFrame."""
        spot_df = pd.read_csv("./data/WTI_Spot.csv")
        forward_df = pd.read_csv("./data/WTI_Forward.csv")
        
        self.price_df = pd.merge(spot_df, forward_df, on="Date", how="outer")
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])
        self.price_df.sort_values('Date', inplace=True)

        self.price_df.dropna(inplace=True)
        self.price_df.reset_index(drop=True, inplace=True)
        self.price_df.to_csv("./data/WTI_Combined.csv", index=False)

    def _calculate_spread(self):
        """Calculates the spread between spot and forward prices."""
        self.price_df['Spread'] = self.price_df['Forward_Price'] - self.price_df['Spot_Price']
        plt.plot(self.price_df['Date'], self.price_df['Spread'], label='Spread', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Spread Price')
        plt.title('Spread between Spot and Forward Prices')
        plt.legend()
        plt.show()
        return self.price_df
    
if __name__ == "__main__":
    commodity = Commodity()
    commodity._read_csv()
    spread_df = commodity._calculate_spread()
    print(spread_df.head())