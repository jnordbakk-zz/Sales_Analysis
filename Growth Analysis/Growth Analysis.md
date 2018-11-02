

```python
## Import dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import files and convert to pandas df
file = os.path.join('ACCT.csv')
accounts_df = pd.read_csv(file, encoding="ISO-8859-1")

file = os.path.join('Product.csv')
product_df = pd.read_csv(file, encoding="ISO-8859-1")

file = os.path.join('OPPTY_metadata.csv')
oppt_df = pd.read_csv(file, encoding="ISO-8859-1")
oppt_df = oppt_df.rename(columns={'ACCT ID': 'ACCT_ID'})
```


```python
# Join the three tables together to create a master file
result = pd.merge(oppt_df, accounts_df, how='left', on=['ACCT_ID'])
mastertable = pd.merge(result, product_df, how='left', on=['Oppty ID'])
mastertable.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Oppty ID</th>
      <th>Oppty Type Summary</th>
      <th>Flip Date</th>
      <th>Close Date</th>
      <th>Amount</th>
      <th>Oppty.Source.Org</th>
      <th>Push Counter</th>
      <th>ACCT_ID</th>
      <th>Fiscal Close QTR (FYYYQ)</th>
      <th>Won</th>
      <th>ACCT_TYP</th>
      <th>LCK_EMP_CNT</th>
      <th>SECTOR_NM</th>
      <th>SEGMENT</th>
      <th>APM Level 1</th>
      <th>APM Level 2</th>
      <th>APM Level 3</th>
      <th>Product Share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0060M000011ka1E</td>
      <td>New Business</td>
      <td>4/10/2016</td>
      <td>4/28/2016</td>
      <td>6642.00</td>
      <td>ECS</td>
      <td>0</td>
      <td>0013000001MsiOW</td>
      <td>FY17Q1</td>
      <td>1</td>
      <td>Unlimited Customer</td>
      <td>17992.0</td>
      <td>Pharma</td>
      <td>Healthcare</td>
      <td>Salesforce Platform</td>
      <td>Lightning Platform</td>
      <td>Core Lightning Platform</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0060M000011ka1n</td>
      <td>Add-On/Upgrade</td>
      <td>4/10/2016</td>
      <td>4/10/2016</td>
      <td>1377.00</td>
      <td>AE</td>
      <td>0</td>
      <td>0013000000hLedR</td>
      <td>FY17Q1</td>
      <td>1</td>
      <td>Enterprise Customer</td>
      <td>620.0</td>
      <td>Agriculture</td>
      <td>Core Mid Market</td>
      <td>Core Premier (A la Carte) &amp; Priority</td>
      <td>Core Premier (A la Carte)</td>
      <td>Core Premier (A la Carte)</td>
      <td>0.245098</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0060M000011ka1n</td>
      <td>Add-On/Upgrade</td>
      <td>4/10/2016</td>
      <td>4/10/2016</td>
      <td>1377.00</td>
      <td>AE</td>
      <td>0</td>
      <td>0013000000hLedR</td>
      <td>FY17Q1</td>
      <td>1</td>
      <td>Enterprise Customer</td>
      <td>620.0</td>
      <td>Agriculture</td>
      <td>Core Mid Market</td>
      <td>Sales Cloud</td>
      <td>Core Sales (SFA)</td>
      <td>SFA</td>
      <td>0.754902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0060M000011ka1Y</td>
      <td>Add-On/Upgrade</td>
      <td>4/11/2016</td>
      <td>4/19/2016</td>
      <td>10150.38</td>
      <td>AE</td>
      <td>0</td>
      <td>0013000000BoVX2</td>
      <td>FY17Q1</td>
      <td>1</td>
      <td>Unlimited Customer</td>
      <td>87.0</td>
      <td>Software and Services</td>
      <td>Core Small Business</td>
      <td>Sales Cloud</td>
      <td>Core Sales (SFA)</td>
      <td>SFA</td>
      <td>0.867000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0060M000011ka1Y</td>
      <td>Add-On/Upgrade</td>
      <td>4/11/2016</td>
      <td>4/19/2016</td>
      <td>10150.38</td>
      <td>AE</td>
      <td>0</td>
      <td>0013000000BoVX2</td>
      <td>FY17Q1</td>
      <td>1</td>
      <td>Unlimited Customer</td>
      <td>87.0</td>
      <td>Software and Services</td>
      <td>Core Small Business</td>
      <td>Sales Cloud</td>
      <td>Data.com</td>
      <td>Data.com</td>
      <td>0.133000</td>
    </tr>
  </tbody>
</table>
</div>



Lets look at customer portfolio of SFDC as a whole


```python
# Group by Segment 
table= mastertable[mastertable['Won'] == 1]
table=mastertable[mastertable['Fiscal Close QTR (FYYYQ)'] < 'FY18Q4']
grouped = table.groupby('SEGMENT')

count=grouped.count()
count.reset_index(level=0, inplace=True)

# Set x axis and tick locations
x_axis = np.arange(len(count))

plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["SEGMENT"], rotation="vertical")

# Set x and y limits
#plt.xlim(-0.75, len(x_axis))
#plt.ylim(0, max(count["Oppty ID"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Number of Wins for All Products by Account Segment before Essential's Launch")
plt.xlabel("Account Segment")
plt.ylabel("Wins")
```




    Text(0,0.5,'Wins')




![png](output_3_1.png)



```python
# Group by Employee count 
table=mastertable.loc[mastertable['SEGMENT'].isin(['Core Small Business'])]
grouped = table.groupby('LCK_EMP_CNT')
count=grouped.count()
count.reset_index(level=0, inplace=True)
count= count[count['Oppty ID'] > 5000]

#Plot data
# Set x axis and tick locations
x_axis = np.arange(len(count))

plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["LCK_EMP_CNT"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
plt.ylim(0, max(count["Oppty ID"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Customer Profile in Core Small Business Segment")
plt.xlabel("# of Employees")
plt.ylabel("# of Accounts")
```




    Text(0,0.5,'# of Accounts')




![png](output_4_1.png)



```python
table=table.loc[mastertable['SEGMENT'].isin(['Core Small Business','Core Mid Market'])]
grouped = table.groupby('APM Level 3')

count=grouped.count()
count.reset_index(level=0, inplace=True)

# Set x axis and tick locations
x_axis = np.arange(len(count))

plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["APM Level 3"], rotation="vertical")

# Set x and y limits
#plt.xlim(-0.75, len(x_axis))
#plt.ylim(0, max(count["Oppty ID"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Number of Wins by Product Type for SMB")
plt.xlabel("Product Type")
plt.ylabel("Wins")
```




    Text(0,0.5,'Wins')




![png](output_5_1.png)


Salesforce Essentials was launched FY18Q4. The functionality of this product is far more basic than other editions and offered at a lower price.--> Meets needs of SMB. Lets take a look at who has actually been buying the product since the release date


```python
#Filter table by Essentials product
table=mastertable.loc[mastertable['APM Level 2'].isin(['Sales Cloud Essentials','Service Cloud Essentials'])]

# Group by Segment 
grouped = table.groupby('SEGMENT')
count=grouped.count()
count.reset_index(level=0, inplace=True)

##Plot the data

# Set x axis and tick locations
x_axis = np.arange(len(count))

plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["SEGMENT"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
plt.ylim(0, max(count["Oppty ID"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Number of Wins for Essential Products by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Win")
```




    Text(0,0.5,'Win')




![png](output_7_1.png)


As expected, small Businesses are the biggest customer. 

Lets look into how many new customers were acquired since Essentials’ launch was in November, 2017 FY18Q4 and how the product is doing compared to Salesforce's other products.


```python

# Quarter of release
table= mastertable[mastertable['Fiscal Close QTR (FYYYQ)'] == 'FY18Q4']

# Group by product and count
grouped = table.groupby('APM Level 2')

count2=grouped.count()
count2.reset_index(level=0, inplace=True)

# Quater after release
table2= mastertable[mastertable['Fiscal Close QTR (FYYYQ)'] == 'FY19Q1']

# Group by product and count
# bar graph for each product
grouped = table2.groupby('APM Level 2')

count3=grouped.count()
count3.reset_index(level=0, inplace=True)

#Quarter before release
table= mastertable[mastertable['Fiscal Close QTR (FYYYQ)'] == 'FY19Q2']

# Group by product and count
grouped = table.groupby('APM Level 2')

count1=grouped.count()
count1.reset_index(level=0, inplace=True)
```


```python
plt.plot(np.arange(len(count2)), count2["ACCT_ID"], color='green', alpha=0.5, label="FY18Q4")
plt.xticks(np.arange(len(count2)), count2["APM Level 2"], rotation="vertical")

plt.plot(np.arange(len(count3)), count3["ACCT_ID"], color='darkblue', alpha=0.5, label="FY19Q1")
plt.xticks(np.arange(len(count3)), count3["APM Level 2"], rotation="vertical")

plt.plot(np.arange(len(count1)), count1["ACCT_ID"], color='dodgerblue', alpha=0.5, label="FY19Q2")
plt.xticks(np.arange(len(count1)), count1["APM Level 2"], rotation="vertical")

plt.style.use('seaborn')

# Set a Title and labels
plt.title("Number of Wins FY18Q3-FY19Q1")
plt.xlabel("Product Type")
plt.ylabel("Wins")

plt.legend()
```




    <matplotlib.legend.Legend at 0x18a0135a630>




![png](output_10_1.png)


One of the identified risks, was that instead of winning new SMB customers, the launch of Essentials would canabalize the SMB Account Executives sales. Let's take a closer look at those sales. 


```python
### Select all Essential wins and vizualize the Oppt Type

table=mastertable.loc[mastertable['APM Level 2'].isin(['Sales Cloud Essentials','Service Cloud Essentials'])]
grouped = table.groupby('Oppty Type Summary')

count=grouped.count()
count.reset_index(level=0, inplace=True)


# Set x axis and tick locations
x_axis = np.arange(len(count))


plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["Oppty Type Summary"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
#plt.ylim(0, max(count["Oppty Type Summary"])+5)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Existing accounts that ugraded or added Essentials to Portfolio")
plt.xlabel("# of accounts")
plt.ylabel("# of new Wins")
```




    Text(0,0.5,'# of new Wins')




![png](output_12_1.png)


Most seem to be "New Business". Let's take closer look at the "Add-On/Upgrade"


```python
grouped = table5.groupby('APM Level 2_BeforeRelease')

count=grouped.count()
count.reset_index(level=0, inplace=True)
count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>APM Level 2_BeforeRelease</th>
      <th>Oppty ID_EssentialSales</th>
      <th>Oppty Type Summary_EssentialSales</th>
      <th>Flip Date_EssentialSales</th>
      <th>Close Date_EssentialSales</th>
      <th>Amount_EssentialSales</th>
      <th>Oppty.Source.Org_EssentialSales</th>
      <th>Push Counter_EssentialSales</th>
      <th>ACCT_ID</th>
      <th>Fiscal Close QTR (FYYYQ)_EssentialSales</th>
      <th>...</th>
      <th>Push Counter_BeforeRelease</th>
      <th>Fiscal Close QTR (FYYYQ)_BeforeRelease</th>
      <th>Won_BeforeRelease</th>
      <th>ACCT_TYP_BeforeRelease</th>
      <th>LCK_EMP_CNT_BeforeRelease</th>
      <th>SECTOR_NM_BeforeRelease</th>
      <th>SEGMENT_BeforeRelease</th>
      <th>APM Level 1_BeforeRelease</th>
      <th>APM Level 3_BeforeRelease</th>
      <th>Product Share_BeforeRelease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B2C Commerce</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Core Premier (A la Carte)</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Core Sales (SFA)</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>...</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>38</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Core Service</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lightning Platform</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Messaging/Journeys</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Quip</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>...</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sales Cloud Einstein</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 35 columns</p>
</div>




```python
(41/(1+2+41+2+1+1+7+2))*100
```




    71.9298245614035




```python
### Select all Essential wins that are not tagged as Oppty Type Summary= "New Business"
table3= table[table['Oppty Type Summary'] != 'New Business']

### now see if any of these accounts existed before launch  FY18Q4 with  different product
table4=mastertable[mastertable['Fiscal Close QTR (FYYYQ)'] < 'FY18Q4']

# Inner join to select the accounts that existed prior to Essential Launch and were tagged as "Add-On/Upgrade
table5=pd.merge(table3,table4, how='inner', on=['ACCT_ID'],suffixes=("_EssentialSales","_BeforeRelease"))

grouped = table5.groupby('APM Level 2_BeforeRelease')

count=grouped.count()
count.reset_index(level=0, inplace=True)

# Set x axis and tick locations
x_axis = np.arange(len(count))


plt.bar(x_axis, count["Oppty ID_EssentialSales"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["APM Level 2_BeforeRelease"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
plt.ylim(0, max(count["Oppty ID_EssentialSales"])+5)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Existing accounts that ugraded or added Essentials to Portfolio")
plt.xlabel("# of accounts")
plt.ylabel("# of new Wins")
```




    Text(0,0.5,'# of new Wins')




![png](output_16_1.png)



```python
## Looking at % of core sales add ons

# find all accounts with core sales opporunities
table6=mastertable[mastertable['APM Level 2'] == 'Core Sales (SFA)']

# find all accounts that had a "Add On"
table7=mastertable[mastertable['Oppty Type Summary'] == 'Add-On/Upgrade']

# do inner join to find which accounts are in both

table8=pd.merge(table6,table7, how='left', on=['ACCT_ID'])
# join back to master data to get all opportunities for those accounts

table9=pd.merge(table8,mastertable, how='left', on=['ACCT_ID'])
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-23-b5df2f4037a1> in <module>()
         12 # join back to master data to get all opportunities for those accounts
         13 
    ---> 14 table9=pd.merge(table8,mastertable, how='left', on=['ACCT_ID'])
    

    /anaconda3/lib/python3.6/site-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
         56                          copy=copy, indicator=indicator,
         57                          validate=validate)
    ---> 58     return op.get_result()
         59 
         60 


    /anaconda3/lib/python3.6/site-packages/pandas/core/reshape/merge.py in get_result(self)
        594             [(ldata, lindexers), (rdata, rindexers)],
        595             axes=[llabels.append(rlabels), join_index],
    --> 596             concat_axis=0, copy=self.copy)
        597 
        598         typ = self.left._constructor


    /anaconda3/lib/python3.6/site-packages/pandas/core/internals.py in concatenate_block_managers(mgrs_indexers, axes, concat_axis, copy)
       5201         else:
       5202             b = make_block(
    -> 5203                 concatenate_join_units(join_units, concat_axis, copy=copy),
       5204                 placement=placement)
       5205         blocks.append(b)


    /anaconda3/lib/python3.6/site-packages/pandas/core/internals.py in concatenate_join_units(join_units, concat_axis, copy)
       5330     to_concat = [ju.get_reindexed_values(empty_dtype=empty_dtype,
       5331                                          upcasted_na=upcasted_na)
    -> 5332                  for ju in join_units]
       5333 
       5334     if len(to_concat) == 1:


    /anaconda3/lib/python3.6/site-packages/pandas/core/internals.py in <listcomp>(.0)
       5330     to_concat = [ju.get_reindexed_values(empty_dtype=empty_dtype,
       5331                                          upcasted_na=upcasted_na)
    -> 5332                  for ju in join_units]
       5333 
       5334     if len(to_concat) == 1:


    /anaconda3/lib/python3.6/site-packages/pandas/core/internals.py in get_reindexed_values(self, empty_dtype, upcasted_na)
       5630             for ax, indexer in self.indexers.items():
       5631                 values = algos.take_nd(values, indexer, axis=ax,
    -> 5632                                        fill_value=fill_value)
       5633 
       5634         return values


    /anaconda3/lib/python3.6/site-packages/pandas/core/algorithms.py in take_nd(arr, indexer, axis, out, fill_value, mask_info, allow_fill)
       1381     func = _get_take_nd_function(arr.ndim, arr.dtype, out.dtype, axis=axis,
       1382                                  mask_info=mask_info)
    -> 1383     func(arr, indexer, out, fill_value)
       1384 
       1385     if flip_order:


    KeyboardInterrupt: 



```python

table9.count()
```


```python
table9["ACCT_ID"].nunique()
```




    0.026198417871154903




```python
table6=mastertable[mastertable['APM Level 2'] == 'Core Sales (SFA)']
table6["Oppty ID"].count()
```




    156498



What is the cost of customer acquistion of Essential? To answer that lets look at sales channel for all prodcts. You can see from graph below, mostly from Account Executives


```python
grouped = mastertable.groupby('Oppty.Source.Org')

count=grouped.count()
count.reset_index(level=0, inplace=True)

# Set x axis and tick locations
x_axis = np.arange(len(count))


plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["Oppty.Source.Org"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
# plt.ylim(0, max(count["Oppty.Source.Org"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Sales Channels of Overall Products wins")
plt.xlabel("Sales Channels")
plt.ylabel("# of new Wins")
```




    Text(0,0.5,'# of new Wins')




![png](output_22_1.png)


Looking at just Essential product in comparison


```python
# Group by Oppty.Source.Org 
table=mastertable.loc[mastertable['APM Level 2'].isin(['Sales Cloud Essentials','Service Cloud Essentials'])]
table.head()

grouped = table.groupby('Oppty.Source.Org')

count=grouped.count()
count.reset_index(level=0, inplace=True)

# Set x axis and tick locations
x_axis = np.arange(len(count))


plt.bar(x_axis, count["Oppty ID"], color='dodgerblue', alpha=0.5)
plt.xticks(x_axis, count["Oppty.Source.Org"], rotation="vertical")

# Set x and y limits
plt.xlim(-0.75, len(x_axis))
# plt.ylim(0, max(count["Oppty.Source.Org"])+500)
# plt.tight_layout()
plt.style.use('seaborn')

# Set a Title and labels
plt.title("Sales Channels of Essential Products wins")
plt.xlabel("Sales Channels")
plt.ylabel("# of new Wins")
```




    Text(0,0.5,'# of new Wins')




![png](output_24_1.png)

