import pandas as pd
import numpy as np
from sunpy.coordinates import sun
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from astropy.time import Time
from datetime import datetime


def alpha(w):
    return (2.1349 * 10**(19))*np.exp((-0.14052)*w)
def beta(w):
    return (16.407 - 0.085284 * w) + (0.00011522 * (w**2))
def delta(w, omega, P):
    return ((-1) * alpha(w) * (omega/1000) + (-1)*(beta(w)) * (P/1013.25))
def miu(r, lattitude, z):
    h = 26-0.1*lattitude
    R = 6371
    r = r/1000
    v = ((R+r)**2)/((R+h)**2)
    k = np.sin(z)**2
    return (1/(1-v*k)**0.5)

# Import and pre-process

file_path = 'data/data_output_jan14_2.csv'
df = pd.read_csv(file_path, header=0)
df = df.dropna()
df = df.reset_index()
df = df.apply(pd.to_numeric, errors='ignore')
df["SZA"] = np.deg2rad(df["SZA"])
file_path = 'Reference Data/Pandora243s1_Toronto-CNTower_L2_rout2p1-8.txt'
data = []
with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
    for i, line in enumerate(file):
        row = line.strip().split(" ")
        if len(row) != 52:
            continue
        data.append(row)

rf = pd.DataFrame(data)  # Adjust column names
rf[0] = pd.to_datetime(rf[0], format='%Y%m%dT%H%M%S.%fZ')
rf[0] = rf[0].dt.strftime('%Y-%m-%d %H:%M:%S')

rf[0] = pd.to_datetime(rf[0])
rf['Date'] = rf[0].dt.date
target_date = pd.to_datetime("2025-01-14").date()
rf = rf[rf['Date'] == target_date]
rf = rf.reset_index(drop=True)

rf = rf[1:].apply(pd.to_numeric, errors='ignore')  # Convert all columns to numeric

df_3bands = {
    'w': [302.432, 307.848, 313.017],
    'w_standard':[305, 312, 320]
}
df_3bands = pd.DataFrame(df_3bands)


df_3bands['w_offset'] = abs(df_3bands["w_standard"] - df_3bands["w"])

### calculate standard delta
P = 999.5
Omega = (rf[38].mean()) * 2241.3986
df_3bands["standard_delta"] = df_3bands["w"].apply(lambda x: delta(x, Omega, P))

print(Omega)

# calculate weighted air mass - miu

df['miu'] = df.apply(lambda row: miu(row['ALTITUDE'], row['LATITUDE'], row['SZA']), axis=1)
df['m'] = (1/np.cos(df['SZA'])) - 0.0018167 * ((1/np.cos(df['SZA']))-1) - 0.002875 * ((1/np.cos(df['SZA']))-1)**2 - 0.0008083 * ((1/np.cos(df['SZA']))-1)**3 

'''
z = np.linspace(60, 75, 1000)
z = np.deg2rad(z)
plt.plot(z, 1/np.cos(z) - 0.0018167 * ((1/np.cos(z))-1) - 0.002875 * ((1/np.cos(z))-1)**2 - 0.0008083 * ((1/np.cos(z))-1)**3, label='m')
y = []
for i in z:
    y.append(miu(70, 1, i))
plt.plot(z, y, label='miu')
'''

# calculate weighted air mass - omega
df_3bands["alpha"] = df_3bands["w"].apply(lambda x: alpha(x))
df_3bands["beta"] = df_3bands["w"].apply(lambda x: beta(x))
df["a_305"] = -df_3bands["alpha"][0] * df["OZONE"]/1000
df["a_312"] = -df_3bands["alpha"][1] * df["OZONE"]/1000
df["a_320"] = -df_3bands["alpha"][2] * df["OZONE"]/1000
df["b_305"] = -df_3bands["beta"][0] * (df["PRESSURE"]/1013.25)
df["b_312"] = -df_3bands["beta"][1] * (df["PRESSURE"]/1013.25)
df["b_320"] = -df_3bands["beta"][2] * (df["PRESSURE"]/1013.25)
df["omega_305"] = (df["a_305"] * df["miu"] + df["b_305"] * df["m"])/(df["a_305"] + df["b_305"])
df["omega_312"] = (df["a_312"] * df["miu"] + df["b_312"] * df["m"])/(df["a_312"] + df["b_312"])
df["omega_320"] = (df["a_320"] * df["miu"] + df["b_320"] * df["m"])/(df["a_320"] + df["b_320"])

# regression between I and omega
df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df['EARTH_ORBIT_FACTOR'] = df['DATETIME'].apply(
    lambda x: (sun.earth_distance(x).value) ** 2
) #F

k = 1
df["LN_AVG_SIG305"] = np.log(df['SIG305']/df['EARTH_ORBIT_FACTOR']) * k
df["LN_AVG_SIG312"] = np.log(df['SIG312']/df['EARTH_ORBIT_FACTOR']) * k
df["LN_AVG_SIG320"] = np.log(df['SIG320']/df['EARTH_ORBIT_FACTOR']) * k

df_3bands["lin_reg_slope"] = df_3bands["w"]
df_3bands["lin_reg_intercept"] = df_3bands["w"]
df_3bands["a+b"] = df_3bands["w"]

df_3bands["lin_reg_slope"][0], df_3bands["lin_reg_intercept"][0] = np.polyfit(df["omega_305"],df['LN_AVG_SIG305'], 1)
df_3bands["lin_reg_slope"][1], df_3bands["lin_reg_intercept"][1] = np.polyfit(df["omega_312"], df["LN_AVG_SIG312"], 1)
df_3bands["lin_reg_slope"][2], df_3bands["lin_reg_intercept"][2] = np.polyfit(df["omega_320"], df["LN_AVG_SIG320"], 1)

df_3bands["a+b"][0] = df["a_305"].mean() + df["b_305"].mean()
df_3bands["a+b"][1] = df["a_312"].mean() + df["b_312"].mean()
df_3bands["a+b"][2] = df["a_320"].mean() + df["b_320"].mean()


plt.scatter(df["omega_305"], df['LN_AVG_SIG305'], label="305")
plt.scatter(df["omega_312"], df['LN_AVG_SIG312'], label="312")
plt.scatter(df["omega_320"], df['LN_AVG_SIG320'], label="320")
plt.legend()
#plt.show()

print(df[["a_305", "b_305","SIG305", "LN_AVG_SIG305", "omega_305", 'm', 'miu']].head())

print(df_3bands)


A1 = alpha(305) - alpha(312)
A2 = alpha(312) - alpha(320)
B1 = beta(305) - beta(312)
B2 = beta(312) - beta(320)

print(A1, A2, B1, B2)

#df.to_csv("output_file.csv", index=False)


#plt.plot(df['SZA'], df["miu"])
#plt.plot(df['SZA'], df["m"])
plt.legend()
#plt.show()

'''
1-14
df_3bands = {
    'w': [306.2545, 308.8587, 318.],
}
'''