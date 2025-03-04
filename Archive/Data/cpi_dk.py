import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
cpi=pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/cpi_dk.xlsx", parse_dates=[0], index_col=0)

"""#Plot the Consumer Price Index
plt.figure(figsize=(10,6))
plt.plot(cpi,label='CPI')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.title('Consumer Price Index')
plt.show()"""


#Change from index to percentage change
inflation=np.log(cpi)
inflation=inflation.diff().dropna()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#Plot ACF and PACF for the original series (if stationary) or the differenced series
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plot_acf(cpi, lags=20, ax=ax[0])
ax[0].set_title('ACF of Differenced Series')

plot_pacf(cpi, lags=20, ax=ax[1])
ax[1].set_title('PACF of Differenced Series')

plt.tight_layout()
plt.show()

