import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Outlier Treatment
df = pd.read_csv(r"C:\notes\projectfedex\fedex.csv")
df.dtypes
df.info()

duplicate = df.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate


# Range
range = max(df.DayofMonth) - min(df.DayofMonth)
range

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Delivery_Status'].quantile(0.75) - df['Delivery_Status'].quantile(0.25)
IQR
lower_limit = df['Delivery_Status'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Delivery_Status'].quantile(0.75) + (IQR * 1.5)

#Mising Values
df = pd.read_csv(r"C:\notes\projectfedex\fedex.csv")
df1=df.isna().sum()

# Trimming Technique
# Let's flag the outliers in the dataset

outliers_df = np.where(df.Delivery_Status > upper_limit, True, np.where(df.Delivery_Status < lower_limit, True, False))

# outliers data
df_out = df.loc[outliers_df, ]

df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# Let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Delivery_Status)

# Replace 
# Replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Delivery_Status'] > upper_limit, upper_limit, np.where(df['Delivery_Status'] < lower_limit, lower_limit, df['Delivery_Status'])))
sns.boxplot(df.df_replaced)

# Winsorization 

from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5, 
                          variables = ['Delivery_Status'])

df_s = winsor_iqr.fit_transform(df[['Delivery_Status']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_


#First Business Moment Decision
#Measure of Central Tendency 
print(df.median())
print(df.mode())

#Second Business Moment Decision
#Measure Of Dispersion
# Variance
df.var()
# Standard Deviation
df.std()

#Third Business Moment Decision
#Skewness
df.skew()

#Fourth Business Moment Decision
#Kurtosis
df.kurt()

############### Graphical Representation     ##################3
#Boxplot
sns.boxplot(df.Delivery_Status)
#There is a outlier as per the boxplot its clearly seen.

# Density Plot
sns.kdeplot(df.Shipment_Delay)
sns.kdeplot(df.Delivery_Status)
#The Density plot reflects that he has outlier as it is moved to the sid eof the graph

#Histogram 
plt.hist(df.Delivery_Status)
plt.hist(df.Carrier_Num)
plt.hist(df.Planned_TimeofTravel)
plt.hist(df.Shipment_Delay)
plt.hist(df.Distance))
plt.hist(df.Destination)

#The Histogram reflects that he has outlier as it is moved to the sid eof the graph .  

################## AUTOEDA ###########################################

import sweetviz as sv

s = sv.analyze(df)
s.show_html()

from autoviz.AutoViz_Class import AutoViz_Class

av = AutoViz_Class()
a = av.AutoViz(r"C:\notes\projectfedex\fedex.csv", chart_format = '.html') 

import os
os.getcwd()

a = av.AutoViz(r"C:\notes\projectfedex\fedex.csv", depVar = 'Delivery_status')

# D-Tale

import dtale
import pandas as pd

df = pd.read_csv(r"C:\notes\projectfedex\fedex.csv")

d = dtale.show(df)
d.open_browser()

# Pandas Profiling

from pandas_profiling import ProfileReport 

p = ProfileReport(df)
p
p.to_file("output.html")

#Data Prepartion
import os
os.getcwd()

from dataprep.eda import create_report

report = create_report(df, title = 'My Report')

report.show_browser()

