# Importing Libraries into spyder IDE
import numpy as np
import pandas as pd
 
# Ignore warnings 
import warnings
warnings.filterwarnings('ignore')

#for  Visualisation the data
import matplotlib.pyplot as plt
import seaborn as sns


#for preprocessing  the data
from sklearn.preprocessing import  StandardScaler, LabelEncoder, MinMaxScaler

#  for Modelling Helping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , cross_val_score

# Applying ML Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#Extract Dataset
#Specify the location to the Dataset and Import them.
df = pd.read_csv('diamonds.csv')
print(df)

# How the data looks
df.head()

#Here we can see all the Features of dataset:
#1.Carat : Carat weight of the Diamond.
#2.Cut : Describe cut quality of the diamond; Quality in increasing order Fair, Good, Very Good, Premium, Ideal .
#3.Color : Color of the Diamond;With D being the best and J the worst.
#4. Clarity : Diamond Clarity refers to the absence of the Inclusions and Blemishes.
#(In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
#5.Depth : The Height of a Diamond, measured from the Culet to the table, divided by its average Girdle Diameter.
#6.Table : The Width of the Diamond's Table expressed as a Percentage of its Average Diameter.
#7.Price : the Price of the Diamond.
#X : Length of the Diamond in mm.
#Y : Width of the Diamond in mm.
#Z : Height of the Diamond in mm.
#Qualitative Features (Categorical) : Cut, Color, Clarity.
#Quantitative Features (Numerical) : Carat, Depth , Table , Price , X , Y, Z.

#To print coloumns
column_names = df.columns
print(column_names)
#In this dataset we are given carat,cut,color,clarity,depth,table,price,x,y and z as a column.


#Find Missing Values
missing_values=df.isnull()
print(missing_values)

#So there are no NaN values in our diamond dataset.

# Drop the 'Unnamed: 0' column as we already have Index.
df.drop(['Unnamed: 0'] , axis=1 , inplace=True)
df.head()

 #Calculating rows and columns in dataset.
df.shape
 # So, We have 53,940 rows and 10 columns
 
#To print Top 5 rows
print(df.head(5))

#To print bottom  5 rows
print(df.tail(5 ))

#finding additional information aboout data
df.info()

#Describe/ summarize the data
#The features described in the above data set are:
#1. Count tells us the number of NoN-empty rows in a feature.
#2. Mean tells us the mean value of that feature.
#3. Std tells us the Standard Deviation Value of that feature.
#4. Min tells us the minimum value of that feature.
#5. 25%, 50%, and 75% are the percentile/quartile of each features.
#6. Max tells us the maximum value of that feature.
describe=df.describe().T
print(describe)

df.describe(include=object)


# As We can see Values of X, Y and Z will be zero. It can't be possible.
#It doesn't make any sense to have either of Length or Width or Height to be zero.
#So, we have to find how many values are zero by look at them.
df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]
len(df[(df['x']==0) | (df['y']==0) | (df['z']==0)])

#We can see there are 20 rows with Dimensions 'Zero'.
#We'll Drop them as it seems better choice instead of filling them with any of Mean or Median


#Dropping Rows with Dimensions 'Zero'.
df = df[(df[['x','y','z']] != 0).all(axis=1)]
# Just to Confirm
df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]

#Now the dataset is Nice and Clean. :)

#To print dimension of dataset
print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])

print( df.carat.min() )

print( df.carat.max() )
#As we can see above the diamond can take the values between 0.2 and 5.01 in carat column.


print( df.cut.unique() )

print( df.color.unique() )

print( df.clarity.unique() )

# As We can see:

#->the types of cuts by using the unique() method.It can be 'Ideal', 'Premium', 'Good', 'Very Good' or 'Fair'.
#->the types of colors are 'E' 'I' 'J' 'H' 'F' 'G' 'D'
#->the types of clarity are 'SI2' 'SI1' 'VS1' 'VS2' 'VVS2' 'VVS1' 'I1''IF'

print( df.price.unique())
print( df.x.unique() )
print( df.y.unique() )
print( df.z.unique() )

#clear data
df.depth.unique()

#Visualization Of All Features

#1. Carat
#Carat refers to the Weight of the Stone, not the Size.
#The Weight of a Diamond has the most significant Impact on its Price.
#Since the larger a Stone is, the Rarer it is, one 2 Carat Diamond will be more Expensive than the Total cost of two 1 Carat Diamonds of the Same Quality.
#The carat of a Diamond is often very Important to People when shopping But it is a Mistake to Sacrifice too much quality for sheer size.
#Source : https://diamondlighthouse.com/blog/2014/10/23/how-carat-weight-affects-diamond-price/

# Visualize via kde plots
sns.kdeplot(df['carat'], shade=True , color='r')

#cart Vs price
sns.jointplot(x='carat' , y='price' , data=df , size=5)
#It seems that Carat varies with Price Exponentially.

#2. Cut
#Although the Carat Weight of a Diamond has the Strongest Effect on Prices, the Cut can still Drastically Increase or Decrease its value.
#With a Higher Cut Quality, the Diamond’s Cost per Carat Increases.
#This is because there is a Higher Wastage of the Rough Stone as more Material needs to be Removed in order to achieve better Proportions and Symmetry.
# Source: https://www.lumeradiamonds.com/diamond-education/diamond-cut

sns.factorplot(x='cut', data=df , kind='count' )

#Cut vs Price
sns.factorplot(x='cut', y='price', data=df, kind='box'  )
# Understanding Box Plot :
# The bottom line indicates the min value of Age.
# The upper line indicates the max value.
# The middle line of the box is the median or the 50% percentile.
# The side lines of the box are the 25 and 75 percentiles respectively.
# As a result,Premium Cut on Diamonds as we can see are the most Expensive, followed by Excellent / Very Good Cut.

#3.  Color
#The Color of a Diamond refers to the Tone and Saturation of Color, or the Depth of Color in a Diamond.
#The Color of a Diamond can Range from Colorless to a Yellow or a Faint Brownish Colored hue.
#Colorless Diamonds are Rarer and more Valuable because they appear Whiter and Brighter.
# Source: https://enchanteddiamonds.com/education/understanding-diamond-color
sns.factorplot(x='color', data=df , kind='count' )

#Color vs Price
sns.factorplot(x='color', y='price' , data=df , kind='violin')

#boxplot
plt.figure(figsize=(10,10))
sns.boxplot(x='color',y='price',data=df,palette='winter')

#4. Clarity
#Diamond Clarity refers to the absence of the Inclusions and Blemishes.
#An Inclusion is an Imperfection located within a Diamond. Inclusions can be Cracks or even Small Minerals or Crystals that have formed inside the Diamond.
#Blemishing is a result of utting and polishing process than the environmental conditions in which the diamond was formed. It includes scratches, extra facets etc.
#Source: https://www.diamondmansion.com/blog/understanding-how-diamond-clarity-affects-value/
labels = df.clarity.unique().tolist()
sizes = df.clarity.value_counts().tolist()
colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()

#finding diamond price equality by boxplot
plt.figure(figsize=(10,10))
sns.boxplot(x='clarity',y='price',data=df,palette='winter')

sns.boxplot(x='clarity', y='price', data=df )
# We can see that VS1 and VS2 affect the Diamond's Price equally having quite high Price margin


#5.  Depth
#The Depth of a Diamond is its Height (in millimeters) measured from the Culet to the Table.
#If a Diamond's Depth Percentage is too large or small the Diamond will become Dark in appearance because it will no longer return an Attractive amount of light.
#Source: https://beyond4cs.com/grading/depth-and-table-values/
plt.hist('depth' , data=df , bins=25)

sns.jointplot(x='depth', y='price' , data=df , kind='regplot', size=5)
#We can conclude from the plot that the Price can vary heavily for the same Depth.
#And the Pearson's Correlation shows that there's a slightly inverse relation between the two

#6. Table
#Table is the Width of the Diamond's Table expressed as a Percentage of its Average Diameter.
#If the Table (Upper Flat Facet) is too Large then light will not play off of any of the Crown's angles or facets and will not create the Sparkly Rainbow Colors.
#If it is too Small then the light will get Trapped and that Attention grabbing shaft of light will never come out but will “leak” from other places in the Diamond.
# Source: https://beyond4cs.com/grading/depth-and-table-values/

sns.kdeplot(df['table'] ,shade=True , color='orange')

sns.jointplot(x='table', y='price', data=df , size=5)

#7. Dimensions
#As the Dimensions increases, Obviously the Prices Rises as more and more Natural Resources are Utilised.

sns.kdeplot(df['x'] ,shade=True , color='r' )
sns.kdeplot(df['y'] , shade=True , color='g' )
sns.kdeplot(df['z'] , shade= True , color='b')
plt.xlim(2,10)

#As seen above these cut,clarity,color contains many outliers.


#Skewness Test
from scipy.stats import skew
skew(df["price"])

#From the above, our skew is a positive value(greater than 1) 
#indicating a positive skewness which means that our mean is greater than the median. 
#Let us look at our data's median and mean prices to verify.



from IPython.display import display
display(np.mean(df["price"]))
display(np.median(df["price"]))

df.groupby('clarity').agg(['mean','std'])
df.groupby('cut').agg(['mean','std'])
df.groupby('color').agg(['mean','std'])
#Let us now log-transform this variable and see if the distribution can get any more closer to normal

# Transforming the target variable
target = np.log(df['price'])
print("Skewness: {}".format(target.skew()))
sns.distplot(target)

# Now again examine each of the independent variables
df['carat'].hist()
#We see that most of the diamond carats range from 0.2-1.2
df['cut'].unique()
sns.countplot(x='cut', data=df)
#We can infer that majority of the cuts are of "Ideal" or "Premium" type, whereas there are very few "Fair" cuts in the data.
df['color'].unique()
sns.countplot(x='color', data=df)
#Clarity
df['clarity'].unique()
sns.countplot(df['clarity'])
#Here, we can infer that most of the diamonds have claritites of 'SI1' or 'VS2'

#Feature Selection to building the ML models

#Scaling of all Features
sns.factorplot(data=df , kind='box' , size=7)
#The Values are Distributed over a Small Scale.

#Correlation Between Features

# Correlation Map
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(data=corr, square=True , annot=True, cbar=True,)

#So, we can say that:
#1. Depth is inversely related to Price.
#This is because if a Diamond's Depth percentage is too large or small the Diamond will become 'Dark' in appearance because it will no longer return an Attractive amount of light.
#2. The Price of the Diamond is highly correlated to Carat, and its Dimensions.
#3. The Weight (Carat) of a diamond has the most significant impact on its Price.
#Since, the larger a stone is, the Rarer it is, one 2 carat diamond will be more 'Expensive' than the total cost of two 1 Carat Diamonds of the same Quality.
#4. The Length(x) , Width(y) and Height(z) seems to be higly related to Price and even each other.
#5. Self Relation ie. of a feature to itself is 1 as expected.
#6. Some other Inferences can also be drawn.

#Plotting histograms

df.hist(figsize=(10, 10))
plt.show()

#This revels the distribution of each property. As expected, we see that the data is not normally distributed.
# After all, how can you expect a 1 carat diamond to be priced just at twice the price of a half-carat given all properties remain the same,
# while a 1 carat diamond looks much bigger to the eye when in a ring, or earrings for that matter, than a half carat one.

#plotting pairplot
plt.figure(figsize=(10,6))
sns.pairplot(df)

#The price vs. carat chart also show that there are some outliers in the dataset i.e. few diamonds that are really over priced!
#Now we need to see the distribution of the dataset. We will create a histogram plot for this. First we define the histplot function.

sns.scatterplot(x="table",y="depth",data=df)
#It gives between relationship table and depth

 # correlation with output variable 
cor_target = abs(corr["price"])

# Selecting highly correlated features 
relevent_features = cor_target[cor_target>0.5]
relevent_features

#Find the most valuable clarity of diamond

clarity_list=list( df.clarity.unique() )

average_prices_list=[]

for each in clarity_list:
    x=df[ df['clarity']==each ]
    current_avg=sum( x.price )/len( x )
    average_prices_list.append( current_avg )
    
new_frame3=pd.DataFrame({"Clarity_Of_Diamond":clarity_list,"Average_Price_Of_Diamond":average_prices_list})

y=new_frame3['Average_Price_Of_Diamond'].max()
z=new_frame3[ new_frame3["Average_Price_Of_Diamond"]==y ]

print("Most valuable type of diamond according to clarity is: ",str( z.Clarity_Of_Diamond ).split()[1] )

most_valuable_clarity=str( z.Clarity_Of_Diamond ).split()[1] 
new_frame3
  
#Find the most valuable cut type of diamond 

cut_types_list=list( df.cut.unique() )
#First extract the unique types of cuts

cut_types_list
# To see what are these types

average_prices_list=[] 
# created an empty list of average_prices_list

for each in cut_types_list: 
    x=df[ df['cut']==each ] 
# extract the same type of cut for each type in the list cut_types_list 
    current_avg=sum( x.price )/len( x )
#calculate averages
    average_prices_list.append( current_avg )
# append result to the  average_prices_list
    
new_frame2=pd.DataFrame({"Cut_Type_Of_Diamond":cut_types_list,"Average_Price_Of_Diamond":average_prices_list })
#created a new dataframe which contains Cut_Type_Of_Diamond and Average_Price_Of_Diamond as a column
print( new_frame2 )

y=new_frame2.Average_Price_Of_Diamond.max() 
# find the maximum value in the Average Price Of Diamond Column
z=new_frame2[ new_frame2['Average_Price_Of_Diamond']==y ] 
#extract the name of the value
print("\nMost valuable type of diamond according to Cut Types is ",str( z.Cut_Type_Of_Diamond ).split()[1] )

most_valuable_cutType=str( z.Cut_Type_Of_Diamond ).split()[1] 
new_frame2  

#Find the most valuable color of diamond 
diamond_color=list( df.color.unique() )

diamond_color
# first we extracted the unique colors of diamonds in the dataset 

average_prices_ofEachColor=[] 
# created an empty list of average prices of each color

for each in diamond_color:
    x=df[ df['color']==each ] 
#extract the same color from the dataframe
    current_avg=sum( x.price ) / len( x ) 
#calculate the average by dividing sum to it's length
    average_prices_ofEachColor.append( current_avg ) 
# append list what we've found in the previous row
new_frame1=pd.DataFrame({"Color_Of_Diamond":diamond_color,"Average_Price_Of_Diamond":average_prices_ofEachColor })
#created a new framework which indicates the color of diamond and the average prices of each
print( new_frame1 ) 
#To see what we've found

y=new_frame1.Average_Price_Of_Diamond.max()
# find the maximum value in the Average Price Of Diamond Column

z=new_frame1[ new_frame1['Average_Price_Of_Diamond']==y ]
#extract the name of the value
print("\nMost valuable type of diamond according to Colors is ",str( z.Color_Of_Diamond ).split()[1] )

most_valuable_color=str( z.Color_Of_Diamond ).split()[1] 
new_frame1

#finding valuable diamonds on the bases of new datframes
valuable=df[(df['cut']==most_valuable_cutType ) & (df['color']==most_valuable_color) & (df['clarity']==most_valuable_clarity) ]
z=valuable.carat.max()

valuable[ valuable['carat']==z ]
#According to the average results we've found here' Valuables list of diamonds according to colour,cut and clarity'



#Now, We find most common three cut types of diamonds,colors of diamonds,clarities of diamonds,depths of diamonds

from collections import Counter

color_count=Counter( list( df.color ) )
most_common_colors = color_count.most_common(3)  

cut_count=Counter( list( df.cut) )
most_common_cuts=cut_count.most_common(3)

clarity_count=Counter( list( df.clarity ) )
most_common_clarity=clarity_count.most_common(3)

depth_count=Counter( list( df.depth ) )
most_common_depth=depth_count.most_common(3)

x1,y1 = zip(*most_common_colors)
x1,y1 = list(x1),list(y1)

x2,y2 = zip(*most_common_cuts )
x2,y2 = list(x2),list(y2)

x3,y3 = zip(*most_common_clarity )
x3,y3 = list(x3),list(y3)

x4,y4 = zip(*most_common_depth )
x4,y4 = list(x4),list(y4)

fig, axes = plt.subplots(nrows=2, ncols=2)
 
sns.barplot(ax=axes[0,0],x=x1, y=y1,palette = sns.cubehelix_palette(len(x)))
sns.barplot(ax=axes[0,1],x=x2, y=y2,palette = sns.cubehelix_palette(len(x)))
sns.barplot(ax=axes[1,0],x=x3, y=y3,palette = sns.cubehelix_palette(len(x)))
sns.barplot(ax=axes[1,1],x=x4, y=y4,palette = sns.cubehelix_palette(len(x)))

#In the [0,0] graph we can see the most common color of diamonds is 'G'.
#In the [0,1] graph we can see the most common cut type of diamonds is 'Ideal'.
#In the [1,0] graph we can see the most common clarity of diamonds is 'SI1'.
#In the [1,1] graph we can see the most common depth of diamonds is '61.8'.


#Feature Engineering
#Create New Feature  called 'Volume'

df['volume'] = df['x']*df['y']*df['z'] 
df.head()


plt.figure(figsize=(5,5))
plt.hist( x=df['volume'] , bins=30 ,color='g')
plt.xlabel('Volume in mm^3')
plt.ylabel('Frequency')
plt.title('Distribution of Diamond\'s Volume')
plt.xlim(0,1000)
plt.ylim(0,50000)

sns.jointplot(x='volume', y='price' , data=df, size=5)

#from above graphs It seems that there is Linear Relationship between Price and Volume (x * y * z).


#Now we Drop X, Y, Z from dataset

df.drop(['x','y','z'], axis=1, inplace= True)
df.head()

# Feature Encoding

#Label the Categorical Features with digits to Distinguish.
#As we can't feed String data for Modelling.
#Changing coloumns datatype to numeric

label = LabelEncoder()
df['cut'] = label.fit_transform(df['cut'])
df['color'] = label.fit_transform(df['color'])
df['clarity'] = label.fit_transform(df['clarity'])

df.head()

#Feature Scaling
#Divide the Dataset into Train and Test, So that we can fit the Train for Modelling Algos and Predict on Test.

# Split the data into train and test.
X = df.drop(['price'], axis=1)
y = df['price']

# Applying Feature Scaling 
#( StandardScaler )

#Let try Scaling and Dimensionality reduction for our input datset and check if there is any impact in Algorithm Results.


MinSc = MinMaxScaler()
df[['depth','table']] = MinSc.fit_transform(df[['depth','table']])

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
print(X_test)
print(y_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying



#Applying Modelling Algos
#As our datset is a regression dataset, so we can apply various regression algoirthm to solve our problem 
#suc as: Linear Regression,Decision Tree and Random Forest and many more.


# Collect all R2 Scores.

R2_Scores = []
Accuracies=[]
models = ['Linear Regression' , 'Lasso Regression' , 'AdaBoost Regression' , 'Ridge Regression' , 'GradientBoosting Regression',
          'RandomForest Regression' ,'KNeighbours Regression', 'DecisionTree Regression']



 #1. Linear Regression
lr = LinearRegression()
lr.fit(X_train , y_train)
accuracies1 = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = lr.predict(X_test)
print('')
print('####### Linear Regression #######')

      
print('Score : %.4f' % lr.score(X_test, y_test))
print(accuracies1)

####### Linear Regression #######
#Score : 0.8810
# Cross-validation score: [0.87904039 0.8862125  0.89004319 0.87548948 0.86134578 0.87897727 0.88077709 0.86606806 0.88187096 0.88043861]

#Displaying the difference between the actual and the predicted
df_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output)

#Checking the accuracy of Linear Regression
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)


print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


R2_Scores.append(r2)
Accuracies.append(accuracies1)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies1)


#plotting reult of Actual and Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# Output Linear Regression

#Score  : 0.8810
#MSE    : 1866686.92 
#MAE    : 927.86 
#RMSE   : 1366.27 
#R2     : 0.88


#2.  Lasso Regression

la = Lasso(normalize=True)
la.fit(X_train , y_train)
accuracies2 = cross_val_score(estimator = la, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = la.predict(X_test)
print('')
print('###### Lasso Regression ######')
print('Score : %.4f' % la.score(X_test, y_test))
print(accuracies2)

#Displaying the difference between the actual and the predicted
df_output2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output2)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


R2_Scores.append(r2)
Accuracies.append(accuracies2)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies2)


#plotting reult of Actual and Predicted values
plt.scatter(y_test, la.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()


# Output of Lasso

#Score  : 0.8659
#MSE    : 2104576.37 
#MAE    : 913.94 
#RMSE   : 1450.72 
#R2     : 0.87 


# 3.  AdaBosst Regression

ar = AdaBoostRegressor(n_estimators=1000)
ar.fit(X_train , y_train)
accuracies3 = cross_val_score(estimator = ar, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = ar.predict(X_test)
print('')
print('###### AdaBoost Regression ######')
print('Score : %.4f' % ar.score(X_test, y_test))
print(accuracies3)

#Displaying the difference between the actual and the predicted
df_output3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output3)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)



R2_Scores.append(r2)
Accuracies.append(accuracies3)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims


sns.boxplot(accuracies3)

#plotting result of Actual and Predicted values
plt.scatter(y_test, ar.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# Output Of AdaBosst Regression

#Score  : 0.8776
#MSE    : 1920619.83 
#MAE    : 1075.71 
#RMSE   : 1385.86 
#R2     : 0.88 

# 4.  Ridge Regression
 
rr = Ridge(normalize=True)
rr.fit(X_train , y_train)
accuracies4 = cross_val_score(estimator = rr, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = rr.predict(X_test)
print('')
print('###### Ridge Regression ######')
print('Score : %.4f' % rr.score(X_test, y_test))
print(accuracies4)

#Displaying the difference between the actual and the predicted
df_output4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output4)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)
Accuracies.append(accuracies4)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies4)

#plotting result of Actual and Predicted values

plt.scatter(y_test, rr.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# Output of Ridge Regression

#Score  : 0.7532
#MSE    : 3871846.51 
#MAE    : 1332.23 
#RMSE   : 1967.70 
#R2     : 0.75 

# 5. GradientBoosting Regression

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls',verbose = 1)
gbr.fit(X_train , y_train)
accuracies5 = cross_val_score(estimator = gbr, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = gbr.predict(X_test)
print('')
print('###### Gradient Boosting Regression #######')
print('Score : %.4f' % gbr.score(X_test, y_test))
print(accuracies5)

#Displaying the difference between the actual and the predicted
df_output5 = pd.DataFrame({'Actual': y_test, 'Predicted': gbr.predict(X_test)})
print(df_output5)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


R2_Scores.append(r2)
Accuracies.append(accuracies5)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies5)


#plotting result of Actual and Predicted values

plt.scatter(y_test, gbr.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# Output of GradientBoosting Regression

#Score  : 0.9044
#MSE    : 1499685.59 
#MAE    : 723.55 
#RMSE   : 1224.62 
#R2     : 0.90 


# 6.RandomForest Regression

rf = RandomForestRegressor()
rf.fit(X_train , y_train)
accuracies6 = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = rf.predict(X_test)
print('')
print('###### Random Forest ######')
print('Score : %.4f' % rf.score(X_test, y_test))
print(accuracies6)

#Displaying the difference between the actual and the predicted
df_output6 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output6)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


# output RandomForest Regression
#Score  : 0.9788
#MSE    : 334491.63 
#MAE    : 296.64 
#RMSE   : 578.35 
#R2     : 0.98 

#Model Tuning :
#A model hyperparameter is external configuration of model. 
#They are often tuned for a predictive problem. 
#Grid-search is used to find the optimal hyperparameters for more accurate predictions and estimate model performance on unseen data.
# We tried to enhance our performance score by using grid search cv and passing parameters on Random Forest regressor.

no_of_test=[100]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2'],'max_depth':[3,5,7,10,15],}
rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2', cv=10)
rf.fit(X_train,y_train)
print('Score : %.4f' % rf.score(X_test, y_test))
pred=rf.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)


R2_Scores.append(r2)
Accuracies.append(accuracies6)


#After applying different tuning parameters the score of RandomForest Regression

#Score : 0.9807
#R2     : 0.98 

#Best score after tuning parameters:
rf.best_score_
#Out: 0.9807509749609956

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies6)

#plotting result of Actual and Predicted values

plt.scatter(y_test, rf.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# 7. KNeighbours Regression

knn = KNeighborsRegressor()
knn.fit(X_train , y_train)
accuracies7 = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = knn.predict(X_test)
print('')
print('###### KNeighbours Regression ######')
print('Score : %.4f' % knn.score(X_test, y_test))
print(accuracies7)

#Displaying the difference between the actual and the predicted
df_output7 = pd.DataFrame({'Actual': y_test, 'Predicted': knn.predict(X_test)})
print(df_output7)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

#Output KNeighbours Regression
#Score  : 0.9557
#MSE    : 694324.56 
#MAE    : 445.90 
#RMSE   : 833.26 
#R2     : 0.96

# Tuning Parameters

n_neighbors=[]
for i in range (0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1],'leaf_size':[30],'p':[2]}
knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='r2', cv=10)
knn.fit(X_train,y_train)
print('Score : %.4f' % knn.score(X_test, y_test))
pred=knn.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)


R2_Scores.append(r2)
Accuracies.append(accuracies7)

#After applying different tuning parameters the score of KNeighbours Regression

#Score : 0.9557
#R2     : 0.96 

#Best score after tuning parameters:
knn.best_score_

#Out: 0.9545858943205903


#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies7)

#plotting result of Actual and Predicted values

plt.scatter(y_test, knn.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

# 8. Decision Tree Regression

Dt =DecisionTreeRegressor()
Dt.fit(X_train , y_train)
accuracies8 = cross_val_score(estimator = Dt, X = X_train, y = y_train, cv = 10,verbose = 1)
y_pred = Dt.predict(X_test)
print('')
print('####### Decision Tree Regression #######')

      
print('Score : %.4f' % Dt.score(X_test, y_test))
print(accuracies8)

df_output8 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_output8)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)
Accuracies.append(accuracies8)

#boxplot feeding the series of 10 scores output by cross-validation for each algorthims

sns.boxplot(accuracies8)

#plotting result of Actual and Predicted values

plt.scatter(y_test, Dt.predict(X_test))
plt.xlabel("Actual")
plt.ylabel("Predicted")
x_lim = plt.xlim()
y_lim = plt.ylim()
plt.plot(x_lim, y_lim, "k--")
plt.show()

#Output Decision Tree Regression
#Score : 0.9649
#MSE    : 551189.67 
#MAE    : 365.55 
#RMSE   : 742.42 
#R2     : 0.96 

# Plotting Cross-Validated Predictions  of all ML models
#(scatter plot using y_test and y_pred to demonstrate the alignment of the predictions of all models).

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred )
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# Finally, Comparison between all models:

# R2-Score of Algorithms

compare = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : R2_Scores})
compare.sort_values(by='R2-Scores' ,ascending=False)
 
#               Algorithms       R2-Scores

#5      RandomForest Regression   0.980669
#7      DecisionTree Regression   0.965121
#6       KNeighbours Regression   0.955750
#4  GradientBoosting Regression   0.904423
#0            Linear Regression   0.881034
#2          AdaBoost Regression   0.877597
#1             Lasso Regression   0.865873
#3             Ridge Regression   0.753243

#Random Forest Regressor gives us the highest R2-Score [ 98% ]after applying hyperparameter tuning.


# Visualizing of Best Score of all models :

MLcompare = pd.DataFrame({'Models by Aman': ['Linear Regression' , 'Lasso Regression' , 'AdaBoost Regression' , 'Ridge Regression' , 'GradientBoosting Regression',
          'RandomForest Regression' ,'KNeighbours Regression', 'DecisionTree Regression'],
                        'Score': [ lr.score(X_test, y_test),  la.score(X_test, y_test) ,ar.score(X_test, y_test) ,rr.score(X_test, y_test), gbr.score(X_test, y_test) ,rf.score(X_test, y_test), knn.score(X_test, y_test), Dt.score(X_test, y_test)]})
Comparision = MLcompare.sort_values(by='Score', ascending=False)
Comparision = Comparision.set_index('Score')
print(Comparision)


#                       Models by Aman
#Score                                
#0.980533      RandomForest Regression
#0.964872      DecisionTree Regression
#0.955750       KNeighbours Regression
#0.904423  GradientBoosting Regression
#0.881034            Linear Regression
#0.877597          AdaBoost Regression
#0.865873             Lasso Regression
#0.753243             Ridge Regression

#As a Result, It is clear that Random Forest Regressor gives us the Best accuracies score of [98% using(rf.best_score_)] when applying GridSearchCV using cross-validation.



#Using Barplot to compare results of R2 Scores
sns.barplot(x='R2-Scores' , y='Algorithms' , data= compare)

#Using factorplot to compare results of R2 Scores
sns.factorplot(x='Algorithms', y='R2-Scores' , data=compare, size=5 , aspect=4)


#Using Boxplot to compare  all models results

fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(Accuracies)
ax.set_xticklabels(models)
plt.show()

#The boxplot gives a graphical representation and useful for making quick comparisons.
# This boxplot result showing the spread of accuracy scores across each cross-validation fold.

#****************************************Finally**********************************************************************

# The Random Forest Regressor is best suitable model for Diamond Price Prediction With highest accuracies
# score of (98%) as compare with other models.

#************************************Thank You*************************************************************