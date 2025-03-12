import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Create a Ticker object for ^GSPC
omxc25 = yf.Ticker("^OMXC25")

# Retrieve the maximum available historical data with weekly intervals
c25 = omxc25.history(period="max", interval="1wk")

"""#EXPLORE THE DATA
c25.head()
c25.info()
c25.describe()
c25.isnull().sum()

# Remove timezone information from the datetime index
c25.index = c25.index.tz_localize(None)

# Write the DataFrame to an Excel file
c25.to_excel("c25.xlsx")

#Visulaise the data
plt.figure(figsize=(10,6))
plt.plot(c25.index, c25['Close'],label="OMXC25")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()"""

#Separate data
close = c25['Close']

