#Import the libraries that are required for the implementation of the code operation
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

#Loading the Dataset into program
retaildf= pd.read_csv("D:/datasets/SampleSuperstore.csv")
retaildf.head()
#Data Analysis
retaildf.info()
#to know number of rows and columns
retaildf.shape
#to see summary statistics
retaildf.describe()
#checking the datatype of all variables
retaildf.dtypes
#Checking Null values
retaildf.isnull().sum()
#Droppping Unwanted Features
retaildf.Country.value_counts()
retaildf=retaildf.drop('Country',axis=1)
retaildf.head()
#Category Level Analysis
retaildf['Category'].unique()
#Number of products in each category
retaildf['Category'].value_counts()

#Number of Sub-categories products are divided.
retaildf['Sub-Category'].nunique()
#Number of products in each sub-category
retaildf['Sub-Category'].value_counts()
#Checking Outliers
plt.figure(figsize=[14,10])
plt.subplot(2,2,1)
retaildf['Quantity'].plot(kind='box')
plt.subplot(2,2,2)
retaildf['Profit'].plot(kind='box')
plt.subplot(2,2,3)
retaildf['Sales'].plot(kind='box')
plt.subplot(2,2,4)
retaildf['Discount'].plot(kind='box')
plt.show()
#let's visualize the sub-categories that are distributed under the 'Categories' using a bar chart.
plt.figure(figsize=(16,8))
plt.bar('Sub-Category','Category',data=retaildf,color='red',edgecolor = 'black', width = 0.5)
plt.show()
#Let's have a clearer picture of the Sub-Categories by visualizing the products in a 'Pie-Chart'
plt.figure(figsize=(14,12))
retaildf['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%", shadow = True)
plt.title("Pie Chart for 'Sub-Categories'") 
plt.show()
#Data Visualization and Analysis
sns.pairplot(retaildf)
#Univariate Analysis
sns.countplot(retaildf['Ship Mode'])

plt.title('Popular shipping modes',size=20)
plt.xlabel('\n Shipping mode',size=10)
plt.ylabel('Number of orders',size=10)
plt.xticks(fontsize=15, rotation=30)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(retaildf['Segment'])

plt.title('Popular Segment',size=20)
plt.xlabel('\n Segment',size=10)
plt.ylabel('Number of orders',size=10)
plt.xticks(fontsize=15)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(retaildf['Category'])

plt.title('Popular Category',size=20)
plt.xlabel('\n Category',size=10)
plt.ylabel('Number of orders',size=10)
plt.xticks(fontsize=12)
plt.show()

plt.figure(figsize=(10,7))
sns.countplot(retaildf['Sub-Category'])

plt.title('Popular Sub-Category',size=20)
plt.xlabel('\n Sub-Category',size=10)
plt.ylabel('Number of orders',size=10)
plt.xticks(fontsize=12,rotation=90)
plt.show()

#Bivariate Analysis
#State wise Profit and Sales Analysis
profit =retaildf.groupby('State')['Profit'].sum()
sales =retaildf.groupby('State')['Sales'].sum()
retaildf_state = pd.concat([sales,profit], axis=1)
retaildf_state.sort_values('Sales', inplace =True, ascending=False)
retaildf_state.head()

retaildf_state.plot(kind='bar')
plt.gcf().set_size_inches(14,10)

top_states_sales = retaildf.groupby('State')['Profit'].sum().reset_index().sort_values(by='Profit',ascending=False)
top_states_sales = top_states_sales.head(20)
sns.catplot('State','Profit',data=top_states_sales, kind='bar',aspect=2,height=8,palette='rainbow')
plt.title('Top 20 states with maximum Profit',size=20)
plt.xticks(size=12, rotation=90)
plt.yticks(size=12)
plt.ylabel('Cumulative Profit',size=18)
plt.xlabel('States',size=18)
plt.show()

#City wise Profit and Sales Analysis.
topcities_sales = retaildf.groupby('City')['Sales'].sum().reset_index().sort_values(by='Sales',ascending=False)
topcities_sales = topcities_sales.head(20)
sns.catplot('City','Sales',data=topcities_sales, kind='bar',aspect=2,height=8,palette='gist_rainbow')
plt.title('Top 20 cities with maximum Sale',size=20)
plt.xticks(size=12, rotation=90)
plt.yticks(size=12)
plt.ylabel('Cumulative Sale',size=18)
plt.xlabel('City',size=18)
plt.show()


topcities= retaildf.groupby('City')['Profit'].sum().reset_index().sort_values(by='Profit',ascending=False)
topcities = topcities.head(20)
sns.catplot('City','Profit',data=topcities, kind='bar',aspect=2,height=8,palette='gist_rainbow')
plt.title('Top 20 Profitable cities',size=25)
plt.xticks(size=15, rotation=90)
plt.yticks(size=15)
plt.ylabel('Cumulative profit',size=18)
plt.xlabel('City',size=18)
plt.show()
#Region wise Profit and Sales Analysis.
retaildf_region =((retaildf.groupby("Region")["Profit"].sum()/retaildf.groupby("Region")["Sales"].sum())*100)
plt.pie(retaildf_region.values, labels=retaildf_region.index, autopct='%1.1f%%', shadow=True) 
plt.gcf().set_size_inches(8,8)
plt.gca().set_title("Profit margin for each region")
print(retaildf_region)
plt.show()
#Discount VS [Profit and Sales]
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
sns.lineplot('Discount', 'Profit', data = retaildf, color = 'b', label= 'Discount')
plt.subplot(1,2,2)
sns.lineplot('Discount', 'Sales', data = retaildf, color = 'b', label= 'Discount')
plt.legend()
#Category VS (Quantity and Profit)
retaildf_category_quant = retaildf.groupby("Category")["Quantity"].sum()
plt.pie(retaildf_category_quant.values,labels=retaildf_category_quant.index, autopct='%1.1f%%',shadow=True) 
plt.gcf().set_size_inches(7,7)
plt.gca().set_title("The Commodities sold")
retaildf_category_quant

retaildf_category = retaildf.groupby("Category")["Profit"].sum()
plt.pie(retaildf_category.values,labels=retaildf_category.index, autopct='%1.2f%%', shadow=True) 
plt.gcf().set_size_inches(7,7)
plt.gca().set_title("According to Category wise Profit share")
print(retaildf_category)
plt.show()

#Sub-Category VS (Quantity, Profit and Sales)
topSubcategory = retaildf.groupby('Sub-Category')['Quantity'].sum().reset_index().sort_values(by='Quantity',ascending=False)
topSubcategory.reset_index(drop=True,inplace=True)

sns.catplot('Sub-Category','Quantity',data=topSubcategory,kind='bar',height=8,aspect=2,palette='plasma')
plt.xticks(size=15,rotation=90)
plt.title(' Classified Sub-categories based on Quantity',size=20)
plt.ylabel('Quantities ordered',size=12)
plt.xlabel('Sub-Category',size=12)
plt.show()

topSubcategory1 = retaildf.groupby('Sub-Category')['Profit','Sales'].sum().reset_index().sort_values(by='Profit',ascending=False)
topSubcategory1.reset_index(drop=True,inplace=True)
sns.catplot('Sub-Category','Profit',data=topSubcategory1,kind='bar',height=8,aspect=2,palette='Oranges')
plt.xticks(size=15,rotation=90)
plt.title(' Classified Sub-categories based on Profit',size=20)
plt.ylabel('Profit',size=12)
plt.xlabel('Sub-Category',size=12)
plt.show()

topSubcategory2 = retaildf.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values(by='Sales',ascending=False)
topSubcategory2.reset_index(drop=True,inplace=True)
sns.catplot('Sub-Category','Sales',data=topSubcategory2,kind='bar',height=8,aspect=2,palette='Wistia')
plt.xticks(size=15,rotation=90)
plt.title(' Classified Sub-categories based on Sales',size=20)
plt.ylabel('Sales',size=12)
plt.xlabel('Sub-Category',size=12)
plt.show()

