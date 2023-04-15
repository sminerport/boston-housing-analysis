import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from itertools import combinations
#set output options

params = {'legend.fontsize': 20,
          'figure.figsize' : (15, 5),
         'axes.labelsize'  : 30,
         'axes.titlesize'  : 30,
         'xtick.labelsize' : 25,
         'ytick.labelsize' : 25,
         'lines.markersize': 25,
         'lines.linewidth' : 5}
plt.rcParams.update(params)
pd.options.display.width = 0
pd.options.display.float_format = '{:,.2f}'.format

#load data as a bunch object (bo)
boston_dataset = load_boston()
print(f'Loading boston dataset...')
print(f'Type dataset object: {type(boston_dataset)}')
print(f'Dataset keys: {boston_dataset.keys()}')

# create dataframe
boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# add target column
boston_df['MEDV'] = boston_dataset.target
# apply log transformation
boston_df['LOGLSTAT'] = boston_df['LSTAT'].apply(np.log)

# basic dataset informatoin
rows, cols = boston_df.shape
print(f'Dataset rows: {rows}')
print(f'Dataset cols: {cols} ({cols-1} features and 1 target variable)')
print(f'Total elements in dataset: {boston_df.size}')
print('---------------------------------------------------------------------')
# print column descriptions
lines = boston_dataset.DESCR.splitlines()[11:28]
print('Description of cols:')
print(*lines, sep='\n')
print('---------------------------------------------------------------------')
print('Info:')
print(boston_df.info())
print('---------------------------------------------------------------------')
print(boston_df.describe())
print('---------------------------------------------------------------------')
print('First 5 and last 5 rows of the dataset:')
print(pd.concat([boston_df.head(), boston_df.tail()]))

# graphs

############### heatmap ############### 
mask = np.zeros_like(boston_df.corr())
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(boston_df.corr().round(2), square=True,
                 annot=True, mask=mask, cmap='coolwarm', annot_kws={'size': 20})
ax.set_title('Heatmap of the Boston Housing Dataset',
             fontsize=25)
ax.tick_params(axis='both', labelsize=20)
# color bar object
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

############### LSTAT distribution ###############
g = sns.displot(boston_df['LSTAT'], bins=30)
g.fig.suptitle('Distribution of LSTAT\n(% lower status of the population)',
               fontsize=25)
g.axes[0,0].set_xlabel('LSTAT', fontsize=20)
g.axes[0,0].set_ylabel('Count', fontsize=20)
g.axes[0,0].tick_params(axis='both', labelsize=20)

############### Histogram: all variables ###############
fig, axes = plt.subplots(round(len(boston_df.columns)/3),3,figsize=(20, 40))
i = 0
for triaxis in axes:
    for axis in triaxis:
        if i < len(boston_df.columns):
            boston_df.hist(column = boston_df.columns[i], bins = 100, ax=axis)
            axis.set_title(boston_df.columns[i], fontsize=25)
            axis.tick_params(axis='both', labelsize=20)
            i += 1
fig.suptitle('Histograms for All Variables in the Boston Housing Dataset',
             fontsize=26)

############### Scatterplots ###############
features = ['LSTAT', 'RM']
target = boston_df['MEDV']
fig, axes = plt.subplots(1, len(features), figsize=(10,5))

for col,ax in zip(features, axes.flat):
    x = boston_df[col]
    y = target
    ax.scatter(x,y,marker='o',s=150)
    ax.set_title(col, fontsize=26)
    ax.set_xlabel(col, fontsize=20)
    ax.set_ylabel('MEDV', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
fig.suptitle('Scatterplot showing the LSTAT and RM variables against MEDV',
              fontsize=30)

############### Transformations ###############
fig, axes = plt.subplots(1, 2)
xs = ['LSTAT', 'LOGLSTAT']
colors = ['green', 'red']
x_line_points = [[0,40],[0,4]]
y_line_points = [[30,0],[50,0]]

for ax,x,color,x_lp,y_lp in zip(axes.flat,xs,colors,
                                x_line_points,y_line_points):
    ax.scatter(boston_df[x], boston_df['MEDV'], color=color, s=150)
    ax.set_xlabel(x,fontsize=20)
    ax.set_ylabel('MEDV',fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.plot(x_lp,y_lp)

def rms_error(actual, predicted):
     ' root-mean-squared-error function'
     # lesser values are better(a<b means a is better)
     mse = mean_squared_error(actual, predicted)
     return np.sqrt(mse)
rms_scorer = make_scorer(rms_error)

boston_ftrs = boston_df[['LOGLSTAT', 'RM']]
boston_tgt = boston_df['MEDV']

############### Train-Test-Split ###############
boston_tts = train_test_split(boston_ftrs, boston_tgt,random_state=2021)
(boston_train_ftrs, boston_test_ftrs,
 boston_train_tgt,  boston_test_tgt) = boston_tts
print('---------------------------------------------------------------------')
print('Split the data into training and test features:' )
print(f'Training Features: {boston_train_ftrs.shape}')
print(f'Training Target: {boston_train_tgt.shape}')
print(f'Testing Features: {boston_test_ftrs.shape}')
print(f'Testing Target: {boston_test_tgt.shape}')
print('---------------------------------------------------------------------')
# create models
lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()


# Create models
regressors = {'baseline'        : DummyRegressor(strategy='mean'),
              'lr'              : lr,
              'lasso'           : lasso,
              'ridge'           : ridge}

fig, ax = plt.subplots(1, 1, figsize=(8,4))
scores = {}
for mod_name, model in regressors.items():
     cv_results = cross_val_score(model,
                                  boston_train_ftrs, boston_train_tgt,
                                  scoring = rms_scorer,
                                  cv=10)
     key = mod_name
     scores[key] = [cv_results.mean(), cv_results.std()]
     lbl = f'{mod_name:s} ({cv_results.mean():5.3f})$\pm${cv_results.std():.2f}'
     ax.plot(cv_results, 'o--', label=lbl, markersize=11)
     ax.set_xlabel('CV-Fold #')
     ax.set_ylabel('RMSE')
     ax.legend(bbox_to_anchor=(1.00, 1.00), fancybox=True, shadow=True)

df = pd.DataFrame.from_dict(scores, orient='index').sort_values(0)
df.columns = ['RMSE', 'STD_DEV']
print('Results on training data')
print(df)
y_predicted = lr.fit(boston_train_ftrs, boston_train_tgt).predict(boston_test_ftrs)
#################### Regression Errors #################################
from itertools import permutations
def regression_errors(figsize, predicted, actual, errors='all'):
    ''' figsize -> subplots;
        predicted/actual data -> columns in a DataFrame
        errors -> 'all' or sequence of indices 
    '''
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             sharex=True, sharey=True)
    df = pd.DataFrame({'actual':actual,
                       'predicted': predicted})
    
    for ax, (x,y) in zip(axes, permutations(['actual',
                                                'predicted'])):
        # plot the data as '.'; perfect as y=x line
        ax.plot(df[x], df[y], '.', label='data')
        ax.plot(df['actual'], df['actual'], '-',
               label='perfection')
        ax.legend()
        
        ax.set_xlabel('{} Value'.format(x.capitalize()))
        ax.set_ylabel('{} Value'.format(y.capitalize()))
        ax.set_aspect('equal')
        
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position('right')
    
    # show connecting bars from data to perfect
    # for all or only those specified?
    if errors == 'all':
        errors = range(len(df))
    if errors:
        acts = df.actual.iloc[errors]
        preds = df.predicted.iloc[errors]
        axes[0].vlines(acts, preds, acts, 'r')
        axes[1].hlines(acts, preds, acts, 'r')
        
regression_errors((10, 5), y_predicted, boston_test_tgt, errors=[0,1,2,3,4,
                                                                 122,123,124,125,126])

def regression_residuals(ax, predicted, actual,
                         show_errors=None, right=False):
    ''' figsize -> subplots;
        predicted/actual data -> columns of a DataFrame
        errors -> 'all' or sequence of indices
    '''
    df = pd.DataFrame({'actual':actual,
                       'predicted':predicted})
    df['error'] = df.actual - df.predicted
    ax.plot(df.predicted, df.error, '.')
    ax.plot(df.predicted, np.zeros_like(predicted), '-')
    
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residual')
        
    if show_errors == 'all':
        show_errors = range(len(df))
    if show_errors:
        preds = df.predicted.iloc[show_errors]
        errors = df.error.iloc[show_errors]
        ax.vlines(preds, 0, errors, 'r')
        
fig, ax= plt.subplots(1,1,figsize=(8,8))

regression_residuals(ax, y_predicted, boston_test_tgt, show_errors=[0,1,2,3,4,
                                                                    122,123,124,125,126])
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual')


coeff = list(zip(boston_ftrs, lr.coef_))

df_coeff = pd.DataFrame(coeff)
df_coeff.columns = ['variable', 'coeff']
df_coeff.set_index('variable')
df_intercept = pd.DataFrame([['INTERCEPT', lr.intercept_]])
df_intercept.columns = ['variable', 'coeff']
df_results = df_intercept.append(df_coeff, ignore_index=True)
df_results = df_results.set_index(['variable'])
df_results.index.name=None
df_results
print('---------------------------------------------------------------------')
print('RMSE on hold-out data:')
test_score = np.sqrt(mean_squared_error(boston_test_tgt, y_predicted))
test_results = pd.DataFrame([('lr', test_score)])
test_results.columns = ['pipeline', 'RMSE']
test_results = test_results.set_index('pipeline')
test_results.index.name=None
print(test_results)
print('---------------------------------------------------------------------')
print('Coefficients:')
print(df_results)
print('---------------------------------------------------------------------')
print('Display True vs. Actual Values:')
y_test = pd.DataFrame(boston_test_tgt)
y_test['LSTAT'] = np.exp(boston_test_ftrs.iloc[:,0])
y_test['RM'] = boston_test_ftrs.iloc[:,1]
y_test['PREDS'] = y_predicted
y_test = y_test.rename(columns={'MEDV': 'TRUE'})
y_test = y_test[['LSTAT','RM','PREDS', 'TRUE']]
print(y_test)
plt.show()

