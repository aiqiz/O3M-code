import pandas as pd
import numpy as np
from sunpy.coordinates import sun
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Import and pre-process
file_path = 'data/data_output_jan14_2.csv'
df = pd.read_csv(file_path, header=0)
df = df.dropna()
df = df.reset_index()

# Calculate Air Mass use Mory's paper method
z = np.deg2rad(df['SZA'])
df['AIR_MASS'] = (1/np.cos(z)) - 0.0018167 * ((1/np.cos(z))-1) - 0.002875 * ((1/np.cos(z))-1)**2 - 0.0008083 * ((1/np.cos(z))-1)**3 


# Calculate earth orbit factor F 
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df['EARTH_ORBIT_FACTOR'] = df['DATETIME'].apply(
    lambda x: (sun.earth_distance(x).value) ** 2
) #F


# Calculate Li -> intercept
df["LN_AVG_SIG305"] = np.log(df['SIG305']/df['EARTH_ORBIT_FACTOR'])
df["LN_AVG_SIG312"] = np.log(df['SIG312']/df['EARTH_ORBIT_FACTOR'])
df["LN_AVG_SIG320"] = np.log(df['SIG320']/df['EARTH_ORBIT_FACTOR'])

slope_305, intercept_305 = np.polyfit(df['AIR_MASS'], df["LN_AVG_SIG305"], 1)
slope_312, intercept_312 = np.polyfit(df['AIR_MASS'], df["LN_AVG_SIG312"], 1)
slope_320, intercept_320 = np.polyfit(df['AIR_MASS'], df["LN_AVG_SIG320"], 1)

df['Predicted_305'] = slope_305 * df['AIR_MASS'] + intercept_305
df['Predicted_312'] = slope_312 * df['AIR_MASS'] + intercept_312
df['Predicted_320'] = slope_320 * df['AIR_MASS'] + intercept_320


# Determine goodness of fit - R^2 and Chi^2
r2_305 = (np.corrcoef(df['AIR_MASS'], df["LN_AVG_SIG305"])[0, 1])**2
r2_312 = (np.corrcoef(df['AIR_MASS'], df["LN_AVG_SIG312"])[0, 1])**2
r2_320 = (np.corrcoef(df['AIR_MASS'], df["LN_AVG_SIG320"])[0, 1])**2
chi_squared_305 = np.sum(((df["LN_AVG_SIG305"] -  df["Predicted_305"]) ** 2) / df["Predicted_305"])
chi_squared_312 = np.sum(((df["LN_AVG_SIG312"] -  df["Predicted_312"]) ** 2) / df["Predicted_312"])
chi_squared_320 = np.sum(((df["LN_AVG_SIG320"] -  df["Predicted_320"]) ** 2) / df["Predicted_320"])

print("r2_305:", r2_305, "r2_312:", r2_312, "r2_320:", r2_320)
print("chi_squared_305:", chi_squared_305, "chi_squared_312:", chi_squared_312, "chi_squared_320:", chi_squared_320)

# Plot out the regression result 
plt.scatter(df['AIR_MASS'], df["LN_AVG_SIG305"], color = '#605678', label='Wavelength 305[nm]', marker='x')
plt.plot(df['AIR_MASS'], df['Predicted_305'], color = '#605678', linestyle='-')

plt.scatter(df['AIR_MASS'], df["LN_AVG_SIG312"], color = '#8ABFA3', label='Wavelength 312[nm]', marker='+')
plt.plot(df['AIR_MASS'], df['Predicted_312'], color = '#8ABFA3', linestyle='-')

plt.scatter(df['AIR_MASS'], df["LN_AVG_SIG320"], color = '#FFBF61', label='Wavelength 320[nm]', marker='o')
plt.plot(df['AIR_MASS'], df['Predicted_320'], color = '#FFBF61', linestyle='-')

plt.xlabel("Air Mass (relative)")
plt.ylabel("Logarithm of Average Intensity for Specific Wavelength")
plt.title("Linear Regression for Determine Top of Atmosphere Coeﬀicients")
plt.legend()
plt.show()


L1 = intercept_305 - intercept_312
L2 = intercept_312 - intercept_320
print(intercept_305, intercept_312, intercept_320)
print("L1:", L1, "L2:", L2)

#print(df)


# Calculate Miu -> Actual and vertical path length of radiation through ozone layer
