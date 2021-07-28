# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import libs needed:

# %%
import os
#for exploratory analysis
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
    
import seaborn as sns
import plotly.express as px

#models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

#evaluation tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from time import time


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# %% [markdown]
# ## Reading Data:

# %%
original_data = pd.read_csv('./data/titanic.csv')
data = original_data.copy()
data.head()

# %% [markdown]
# # EDA:

# %%
data.info()


# %%
plt.figure(figsize=(20, 6))
sns.heatmap(data.isnull(),
            yticklabels=False,
            cmap='magma',
            cbar=False)
plt.title('Missing values representation')
plt.show()


# %%
data.isnull().sum()

# %% [markdown]
# ## Handling missing values:

# %%
data['Age'].fillna(data['Age'].mean(), inplace=True)
data[:3]


# %%
cabin = data.groupby(data['Cabin'].isna())['Survived'].count()/data['Survived'].count()
cabin


# %%
data['Cabin'] = np.where(data['Cabin'].isna(), 0, 1)
data.head()

# %% [markdown]
# ## Plotting data:

# %%
sns.pairplot(data, hue='Survived')


# %%
corr = data.corr()
mask = np.triu(corr.corr())
plt.figure(figsize=(18, 15))
sns.heatmap(corr,
            annot=True,
            fmt='.1f',
            cmap='coolwarm_r',
            mask=mask,
            linewidths=1,
            vmin=-1,
            vmax=1)

plt.show()

# %% [markdown]
# ### Exploring categorical data:

# %%
categoricals = data[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']]


def plot_frequency(cat):
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for ax, cat in zip(axes, categoricals):
        plot = categoricals[cat]
        total = len(categoricals[cat])
        
        sns.countplot(plot, palette='pastel', ax=ax)


        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 10,
                    '{:1.2f}%'.format((height / total) * 100),
                    ha="center")

        plt.ylabel('Count')


# %%
plot_frequency(categoricals)


# %%

def plot_survival(categoricals, data):
        
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for ax, cat in zip(axes, categoricals):
        plot = categoricals[cat]
        total = categoricals.groupby
        
        if cat == 'Survived':
            sns.countplot(data[cat], 
            palette='pastel', 
            ax=ax)
        else:
            sns.countplot(plot,
                          data=data,
                          hue='Survived',
                          palette='pastel',
                          ax=ax)
            ax.legend(title='Survived:',
                      loc='upper right',
                      labels=['No', 'Yes'])
        plt.ylabel('Count')


# %%
plot_survival(categoricals, data)

# %% [markdown]
# ### Exploring numerical data:

# %%
def plot_numericals(df, feature):
        
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    grid = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    
    #histogram:
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(data.loc[:, feature],
                 hist=True,
                 kde=True,
                 norm_hist=False,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])
                                                     


    #QQ_plot:
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Probability Plot')
    stats.probplot(data.loc[:, feature],
                   plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    #boxPlot:    
    ax3 = fig.add_subplot(grid[:, 2])   
    ax3.set_title('Box Plot')
    sns.boxplot(data.loc[:, feature], 
                orient='v', 
                ax=ax3, 
                color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{feature}', fontsize=24)


# %%
plot_numericals(data,'Age')
plot_numericals(data,'Fare')


# %%
data.groupby('Pclass')['Fare'].describe()


# %%
fare_buckets = pd.cut(data['Fare'], right=True, bins=[0,10,20,30,40,50,60,70,80,90,100,600])
n_data = data[['Pclass','Survived']].copy()
fare = n_data.assign(fare_buckets=fare_buckets)
#for better viz:
fare['Survived'] = fare['Survived'].map({0:'died', 1:'survived'})
display(fare)


# %%
#survivors per class and fare bucket:

pivot_t= pd.pivot_table(fare, index=['fare_buckets'], columns=['Pclass','Survived'], aggfunc=len, fill_value=0)
pivot_t.style.background_gradient(axis=1)


# %%
#spot the death pattern per fare bucket:

pd.crosstab(fare['fare_buckets'], fare['Survived']).style.background_gradient(axis=1, cmap='YlOrRd')


# %%
def heatmap():
    plt.figure(figsize=(20,12))
    sns.heatmap(pivot_t, annot=True, cmap='OrRd', fmt='g')
    plt.title('Class and fare bucket: Did a labeled ticket saved you? \n\n', fontsize=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=10)
    plt.tick_params(axis='both', which='major', labelsize=12, labelbottom = True, bottom=True, top = True, labeltop=True)
    plt.show()

heatmap()


# %%
for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=data, kind='point', aspect=3)

# %% [markdown]
# ### Merging 'SibSp' & 'Parch' into 'Family' col:

# %%
data['Family_size'] = data['SibSp'] + data['Parch']
data[:5]     

# %% [markdown]
# ### Drop merged cols:

# %%
data.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
data_backup_1 = data.copy()
data[:3]

# %% [markdown]
# ### Dummifing the 'Sex' col:

# %%
sex_type = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_type)
data.head()

# %% [markdown]
# ### Exploring cabin col:
# %% [markdown]
# ### Drop unneeded cols & write out a cleaned version:

# %%
data.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
data[:3]


# %%
data_backup_2 = data.copy()
data.to_csv('./data/cleaned_titanic.csv', index=False)

# %% [markdown]
# ## Splitting data for training:

# %%
features = data.drop(['Survived'], axis=1)
label = data.Survived


# %%
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=42)


# %%
for i in [y_train, y_val, y_test]:
    print(round(len(i)/len(label), 1))

# %% [markdown]
# ## Write out training sub-datasets

# %%
X_train.to_csv('./data/split_X_train.csv', index=False)
X_val.to_csv('./data/split_X_val.csv', index=False)
X_test.to_csv('./data/split_X_test.csv', index=False)

y_train.to_csv('./data/split_y_train.csv', index=False)
y_val.to_csv('./data/split_y_val.csv', index=False)
y_test.to_csv('./data/split_y_test.csv', index=False)

# %% [markdown]
# # Training models
# %% [markdown]
# ## Class: 

# %%
def print_results(model, name):
    print('BEST PARAMS: {}\n'.format(model.best_params_))

    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        
    joblib.dump(model.best_estimator_, './data/{}.pkl'.format(name))

# %% [markdown]
# ## *LogisticReg*:

# %%
lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

model_LR = GridSearchCV(lr, parameters, cv=5)
model_LR.fit(X_train, y_train)

print_results(model_LR, 'model_LR')

# %% [markdown]
# ## *SupportVectorMachine*:
# 

# %%
svc = SVC()
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}

model_SVC = GridSearchCV(svc, parameters, cv=5)
model_SVC.fit(X_train, y_train)

print_results(model_SVC, 'model_SVC')

# %% [markdown]
# ## *MultilayerPerceptron*:

# %%
mlp = MLPClassifier()
parameters = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

model_MLP = GridSearchCV(mlp, parameters, cv=5)
model_MLP.fit(X_train, y_train)

print_results(model_MLP, 'model_MLP')

# %% [markdown]
# ## *RandomForest*:

# %%
rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}

model_RF = GridSearchCV(rf, parameters, cv=5)
model_RF.fit(X_train, y_train)

print_results(model_RF, 'model_RF')

# %% [markdown]
# ## *GradiantBoosting*:

# %%
gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [1, 3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

model_GB = GridSearchCV(gb, parameters, cv=5)
model_GB.fit(X_train, y_train)

print_results(model_GB, 'model_GB')

# %% [markdown]
# ## *DecisionTree*:

# %%
dt = DecisionTreeClassifier()
parameters= {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 3, 5, 7, 9, 10, 100],
    'max_features': [None, 5],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [2,5,10]

}

model_DT = GridSearchCV(dt, parameters, cv=5)
model_DT.fit(X_train, y_train)

print_results(model_DT, 'model_DT')

# %% [markdown]
# # Evaluating the models and electing the best performer: 

# %%
models = {}

for mdl in os.listdir('./data'):
    if mdl.endswith('.pkl'):
        md = mdl[6:-4]
        models[md] = joblib.load('./data/{}'.format(mdl))
models


# %%
def evaluate_model(name, model, X_val, y_val):
    start = time()
    pred = model.predict(X_val)
    end = time()
    accuracy = round(accuracy_score(y_val, pred), 3)
    precision = round(precision_score(y_val, pred), 3)
    recall = round(recall_score(y_val, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                   accuracy,
                                                                                   precision,
                                                                                   recall,
                                                                                   round((end - start)*1000, 1)))


# %%
for name, mdl in models.items():    
    evaluate_model(name, mdl, X_val, y_val)


# %%
def evaluate_model(X_val, y_val):

    row_idx = 0
    table = pd.DataFrame()
    
    for label, model in  models.items():
        start = time()
        pred = model.predict(X_val)
        end = time()
        accuracy = round(accuracy_score(y_val, pred), 3)
        precision = round(precision_score(y_val, pred), 3)
        recall = round(recall_score(y_val, pred), 3)

        table.loc[row_idx, 'Model'] = label
        table.loc[row_idx, 'Accuracy'] = accuracy
        table.loc[row_idx, 'Precision'] = precision
        table.loc[row_idx, 'Recall'] = recall
        table.loc[row_idx, 'Latency'] = round((end - start)*1000, 1)

        table.sort_values(by=['Accuracy'], ascending=False, inplace=True)
        row_idx +=1
    
    return table


# %%
evalu_table = evaluate_model(X_val, y_val)
evalu_table.style.background_gradient(cmap='summer')


