# Redwine-quality-check-in-machine-learning

Hi everyone,
im chaya and i like machine learning with python so i did this project with machine learning

                #importing the  data set
df=pd.read_csv('winequality-red.csv')
data=df.head(10)
print(data)


#checking the null vslue
            #Dara Wrangling
data1=df.isnull()
data2=df.isnull().sum()
print(data1)
print(data2)
sns.heatmap(data.isnull(),yticklabels=False)
plt.title('data wrangling')
plt.show()


#DATA ANALYSIS
#univarate analysis
#min 5 histogram

df.hist(column='sulphates' ,density='False',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of sulphate quantity')
plt.ylabel('range')
plt.xlabel('sulphate quantity')
plt.show()


df.hist(column='fixed acidity' , density='ture',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of fixed acidity quantity')
plt.ylabel('range')
plt.xlabel('fixed acidity quantity')
plt.show()

df.hist(column='alcohol' , density='ture',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of alcohol quantity')
plt.ylabel('range')
plt.xlabel('alcohol quantity')
plt.show()

df.hist(column='density' , density='ture',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of density quantity')
plt.ylabel('range')
plt.xlabel('density quantity')
plt.show()

df.hist(column='volatile acidity' , density='ture',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of volatile acidity quantity')
plt.ylabel('range')
plt.xlabel('volatile quantity')
plt.show()

df.hist(column='pH' , density='ture',color='b',edgecolor='k',alpha=0.6)
plt.title('Histogram of ph')
plt.ylabel('range')
plt.xlabel('ph quantity')
plt.show()


              #bivariate data analysis
                #min 5 SCATTERPLOT

df.plot.scatter(x='alcohol',y='pH')
plt.title('scatterplot for alcohol and PH')
plt.show()
df.plot.scatter(x='volatile acidity',y='density')
plt.title('scatterplot for volatile acidity and density')
plt.show()
df.plot.scatter(x='residual sugar',y='chlorides')
plt.title('scatterplot for residual sugar and chlorides')
plt.show()
df.plot.scatter(x='free sulfur dioxide',y='total sulfur dioxide')
plt.title('scatterplot for  free sulfur  dioxide and total sulfur dioxide')
plt.show()
df.plot.scatter(x='sulphates',y='quality')
plt.title('scatterplot for sulphate and quality')
plt.show()




    #scatter plot by using seaborn

sns.set_style('white')
sns.set_style('ticks')
sns.regplot(x='alcohol',y='pH',color='g',data=data)
plt.show()

sns.lmplot(x='alcohol',y='pH',hue='quality',markers='+',data=data)
plt.show()

    #MODELING
        #logisitic Regression
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews
print(df['Reviews'])
d=df['Reviews'].unique()
print(d)
from collections import Counter

dc=Counter(df['Reviews'])

#Split the x and y variables
x = df.iloc[:,:11]
y = df['Reviews']
xh=x.head(10)
print(xh)
yh=y.head(10)
print(yh)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
print(x_pca)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.title('finding the gird for new column')
plt.show()

pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)
print(x_new)


#spliting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print( lr_acc_score*100)
