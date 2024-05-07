# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```py
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```
![alt text](out1.png)
```py
df=sns.load_dataset("tips")
df
```
![alt text](out2.png)
```py
sns.lineplot(x='total_bill',y='tip',data=df,hue='sex',linestyle='solid',legend='auto')
```
![alt text](out3.png)
```py
x=[1,2,3,4,5]
y1=[3,4,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi-line Plot')
plt.xlabel('X-label')
plt.ylabel('Y-label')
```
![alt text](out4.png)
```py
tips=sns.load_dataset('tips')
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tip=tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8,6))
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill')
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title("Average Total Bill and Tip by Day")
plt.legend()
```
![alt text](out5.png)
```py
avg_total_bill=tips.groupby('time')['total_bill'].mean()
avg_tip=tips.groupby('time')['tip'].mean()
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill',width=0.4)
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip',width=0.4)
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title("Average Total Bill and Tip by Day")
plt.legend()
```
![alt text](out6.png)
```py
years=range(2000,2012)
apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.9372,0.939]
oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]
plt.bar(years,apples)
plt.bar(years,oranges,bottom=apples)
```
![alt text](out7.png)
```py
dt=sns.load_dataset('tips')
sns.barplot(x='day',y='total_bill',hue='sex',data=dt,palette='Set1')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title("Total Bill by Day and Gender")
plt.legend()
```
![alt text](out8.png)
```py
tips=sns.load_dataset('tips')
sns.scatterplot(x='total_bill',y='tip',hue='sex',data=tips)
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
```
![alt text](out9.png)
```py
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var=np.random.randn(1000)
num_var=pd.Series(num_var,name='Numerical Variable')
num_var
```
![alt text](out10.png)
```py
sns.histplot(data=num_var,kde=True)
```
![alt text](out11.png)
```py
np.random.seed(0)
marks=np.random.normal(loc=70,scale=10,size=100)
marks
```
![alt text](out12.png)
```py
sns.histplot(data=marks,bins=15,kde=True,stat='count',cumulative=False,multiple='stack',element='bars',palette='Set2',shrink=1.0)
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of students Marks')
```
![alt text](out13.png)
```py
import seaborn as sns
import pandas as pd
tips=sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue=tips['sex'])
```
![alt text](out14.png)
```py
sns.boxplot(x='day',y='total_bill',hue='smoker',data=tips,linewidth=3,width=0.6,boxprops={'facecolor':'darkblue','edgecolor':'lightblue'},whiskerprops={'color':'black','linestyle':'--','linewidth':1.5},capprops={'color':'black','linestyle':'--','linewidth':1.5})
```
![alt text](out15.png)
```py
sns.violinplot(x='day',y='total_bill',hue='smoker',data=tips,linewidth=2,width=0.6)
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Violin Plot of Total Bill by day and Smoker Status')
```
![alt text](out16.png)
```py
import seaborn as sns
sns.set(style = 'whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x ='day', y ='tip', data = tip,palette='Set2')
```
![alt text](out17.png)
```py
import seaborn as sns
sns.set(style = 'whitegrid')
tip = sns.load_dataset('tips')
sns.violinplot(x=tip["total_bill"])
```
![alt text](out18.png)
```py
import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
sns.violinplot(x="tip", y="day", data=tip,palette='Set3')
```
![alt text](out19.png)
```py
mart=pd.read_csv("/content/titanic_dataset (1).csv")
mart
```
![alt text](out20.png)
```py
mart=mart[['PassengerId', 'Survived', 'Age', 'Name', 'Ticket', 'Embarked']] 
mart.head(10)
```
![alt text](out21.png)
```py
sns.kdeplot(data=mart,x='PassengerId')
```
![alt text](out22.png)
```py
sns.kdeplot(data=mart,x='Age')
```
![alt text](out23.png)
```py
sns.kdeplot(data=mart)
```
![alt text](out24.png)
```py
sns.kdeplot(data=mart,x='PassengerId',hue='Survived',multiple='stack')
```
![alt text](out25.png)
```py
sns.kdeplot(data=mart,x='PassengerId',y='Survived')
```
![alt text](out26.png)
```py
data = np.random.randint(low = 1, high = 100, size = (10,10))
hm=sns.heatmap(data=data,annot=True)
```
![alt text](out27.png)
```py
hm=sns.heatmap(data=data)
```
![alt text](out28.png)
# Result:
Thus, all the data visualization techniques of seaborn has been implemented.
