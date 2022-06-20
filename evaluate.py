from math import sqrt
import matplotlib.pyplot as plt

def plot_residuals(x, y, yhat):
    '''
    Takes in a independent variable set (x) and resulting (actual) dependent variable set (y), 
    along with predictions based on a linear regression model (yhat) and creates two plots:
    (1) a plot of y vs residuals differences between y and yhat);
    (2) a plot of the residuals vs x.
    '''
    residual = y - yhat
    plt.figure(figsize=(12,6))
    # Subplot of residual increasing with y
    plt.subplot(121)
    plt.scatter(y,residual)
    plt.xlabel('Independent Variable')
    plt.ylabel('Residual')
    plt.title('Residual as independent variable increases')
    # Subplot of errors up and down
    plt.subplot(122)
    plt.scatter(x,residual)
    plt.xlabel('Feature')
    plt.ylabel('Residual')
    plt.title('Residual as Feature increases')
    plt.show()

def regression_errors(y,yhat):
    '''
    Returns linear regression errors down to RMSE resulting from the actual values (y)
    and predicted values from the linear regression (yhat)
    '''
    SSE = round(((y-yhat)**2).sum())
    ESS = round(((yhat-y.mean())**2).sum())
    TSS = round(ESS + SSE)
    MSE = round(SSE/y.shape[0])
    RMSE = round(sqrt(MSE))
    print(f'SSE = {SSE}\nESS = {ESS}\nTSS = {TSS}\nMSE = {MSE}\nRMSE = {RMSE}')
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    '''
    Returns baseline errors down to RMSE resulting from the actual values (y)
    and the baseline which is the mean of y.
    '''
    SSE = round(((y-y.mean())**2).sum())
    MSE = round(SSE/y.shape[0])
    RMSE = round(sqrt(MSE))
    print(f'Baseline SSE = {SSE}\nBaseline MSE = {MSE}\nBaseline RMSE = {RMSE}')
    return SSE, MSE, RMSE

def better_than_baseline(y,yhat):
    '''
    Evaluates whether the model is an improvement over the baseline prediction.
    Utilizes the regression errors function and baseline mean functions.
    '''
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y,yhat)
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    if RMSE < RMSE_baseline:
        print('The model performs better than baseline.')
        return True
    else:
        print('The model FAILS to peform better than baseline.')
        return False
