import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def zillow_model_comparator(X_train,y_train,X_validate,y_validate, features):
    '''
    This function creates a DataFrame to compare a number of linear regression models.
    These were done piecemeal in the zillow workbook, and were then cut and pasted in here.
    This is specific to the zillow report.  Note that it takes in the training and validating data.
    '''
    model_comparator = []
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # Ensure only selected features being used
    X_train = X_train[features]
    X_validate = X_validate[features]

    # 1. Predict target_pred_mean
    abs_logerror_pred_mean = y_train['abs_logerror'].mean()
    y_train['abs_logerror_pred_mean'] = abs_logerror_pred_mean
    y_validate['abs_logerror_pred_mean'] = abs_logerror_pred_mean

    # 2. compute target_pred_median
    abs_logerror_pred_median = y_train['abs_logerror'].median()
    y_train['abs_logerror_pred_median'] = abs_logerror_pred_median
    y_validate['abs_logerror_pred_median'] = abs_logerror_pred_median

    # 3. RMSE of target_pred_mean
    rmse_train_mean = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_mean)**(1/2)

    # 4. RMSE of target_pred_median
    rmse_train_median = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_median)**(1/2)
    rmse_validate_median = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_median)**(1/2)

    results = {'Model' : 'Baseline Median','RMSE Train':rmse_train_median,'RMSE Validate':rmse_validate_median}
    model_comparator.append(results)
    results = {'Model' : 'Baseline Mean','RMSE Train':rmse_train_mean,'RMSE Validate':rmse_validate_mean}
    model_comparator.append(results)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.abs_logerror)

    # predict train
    y_train['abs_logerror_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train_ols = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_lm)**(1/2)

    # predict validate
    y_validate['abs_logerror_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate_ols = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_lm)**(1/2)

    results = {'Model' : 'OLS','RMSE Train':rmse_train_ols,'RMSE Validate':rmse_validate_ols}
    model_comparator.append(results)

    # create the model object
    lars = LassoLars(alpha=2.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.abs_logerror)

    # predict train
    y_train['abs_logerror_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train_lassolars = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_lars)**(1/2)

    # predict validate
    y_validate['abs_logerror_pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate_lassolars = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_lars)**(1/2)

    results = {'Model' : 'LassoLars','RMSE Train':rmse_train_lassolars,'RMSE Validate':rmse_validate_lassolars}
    model_comparator.append(results)

    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.abs_logerror)

    # predict train
    y_train['abs_logerror_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train_glm = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_glm)**(1/2)

    # predict validate
    y_validate['abs_logerror_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate_glm = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_glm)**(1/2)

    results = {'Model' : 'GLM','RMSE Train':rmse_train_glm,'RMSE Validate':rmse_validate_glm}
    model_comparator.append(results)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.abs_logerror)

    # predict train
    y_train['abs_logerror_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train_p2 = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_lm2)**(1/2)

    # predict validate
    y_validate['abs_logerror_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate_p2 = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_lm2)**(1/2)

    results = {'Model' : 'Polynomial, Deg. 2','RMSE Train':rmse_train_p2,'RMSE Validate':rmse_validate_p2}
    model_comparator.append(results)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree3 = pf.transform(X_validate)

    # create the model object
    lm3 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm3.fit(X_train_degree3, y_train.abs_logerror)

    # predict train
    y_train['abs_logerror_pred_lm3'] = lm3.predict(X_train_degree3)

    # evaluate: rmse
    rmse_train_p3 = mean_squared_error(y_train.abs_logerror, y_train.abs_logerror_pred_lm3)**(1/2)

    # predict validate
    y_validate['abs_logerror_pred_lm3'] = lm3.predict(X_validate_degree3)

    # evaluate: rmse
    rmse_validate_p3 = mean_squared_error(y_validate.abs_logerror, y_validate.abs_logerror_pred_lm3)**(1/2)

    results = {'Model' : 'Polynomial, Deg. 3','RMSE Train':rmse_train_p3,'RMSE Validate':rmse_validate_p3}
    model_comparator.append(results)

    df_results = pd.DataFrame(model_comparator).round(4).set_index('Model')
    df_results['Overfit %'] = (100*(df_results['RMSE Validate'] - df_results['RMSE Train'])/df_results['RMSE Train']).round(2)  
    return df_results.T

