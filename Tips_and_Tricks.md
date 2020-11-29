# Kaggle tips and tricks
From diferent sources (Analytics Vidhya, Gilberto Titericz, Kaggle Days, ...)

## Exploratory Data Analysis

Check if the problem is a hidden time series problem

Check correlation feature and target


## Feature pre-processing


### Scaling

![Scaling tricks](images/Scaling.png)

Scaling makes difference for linear models and neural networks. For linear models it's necessarry to try different scaling strategies.

A good reason to start with decision trees when working on a new dataset is that they do not need to scale.

![Target scaling](images/Target_scaling.png)

![Target scaling formulas](images/Target_scaling_formulas.png)

![Target scaling code](images/Target_scaling_code.png)

![Allstate competition](images/AllState_competition.png)
In this competition, Gilberto got top 7 rank by creating a different model scaling the target as per the formulas above and ensembling the different models



### NaN imputation

Add Extra column to tell the model that the feature and row was imputed
![NaN imputations](images/NaN_imputation.png)

![Categorical features](images/NaN_imput_categorical.png)

**Mean Target Encoder**
* Replace the category by the mean of 
* Works for classification and regression problems.
* Works better for high cardinality variables
* Must be processed using CV or out-of-fold encoding
![Mean_Target_Encoder](images/Mean_Target_Encoder.png)

A new feature (Kfold_mean_Enc) is created using Mean Target Encoder strategy by using Cross Validation. This approach reduces the overfitting on the test set that simple Mean Target Encoding would cause.
* Split the train set in K folds.
* For each fold the value of Kfold_mean_Enc is the mean of the target values of the categorical common values in the this fold. 
* To generate the Kfold_mean_Enc feature in the test set, the mean value of Kfold_mean_Enc in the trains set per categorical value can be used.

[Mean Target Encoder explained](https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b)


## Feature Engineering

Linear combinations of features (2-way, 3-way)
![feature_transformations](images/Linear_transformations.png)

3-way + 5-fold Mean Target Encoder gives good results in the test set

```
def target_encode_simple (df_train,df_valid,col,target):
    dt = df_train.groupby(col)[target].agg(['mean]).reset_index(drop=False)
    tmp = df_valid.merge(dt,on=col,how='left')['mean'].values
    return tmp

```

Aggregate one numerical feature based on a categorical feature. For example in the Titanic dataset, agregate the age feature by passenger class feature (mean, count, maximum, minim, std, skewness)
train.group(['class'])['Age'].agg(['count','mean','min','max','skew']).reset_index()

In Time Series, for Kaggle competitions, define lag that fit correctly depending on the test set size.


## Feature selection



## Model selection


## Hyperparameter optimization







