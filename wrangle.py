import pandas as pd
import numpy as np
from splitter import splitter
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

def remove_columns(df, cols_to_remove):
    '''
    Removes indicated columns from dataframe
    '''  
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .75, prop_required_row = .75):
    '''
    Removes columns, then rows, containing greater than the require column number of nulls (default = 25%)
    '''
    col_threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=col_threshold, inplace=True)
    row_threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=.75, prop_required_row=.75):
    '''
    Combines the missing values remover with the column dropper
    '''
    df_og = df.copy()
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    print(f'Features removed: \n{[x for x in df_og.columns if x not in df.columns]}')
    df = df.dropna()
    #print(f'----\nDataset reduced from {df_og.shape[0]} to {df.shape[0]} (dropped {df_og.shape[0] - df.shape[0]} rows)\n----\n')
    return df

def outlier_destroyer(df,k,cols_to_remove):
    '''
    Finds the outliers for each column using k * IQR; for many dataframes it's lower bound becomes zero
    Then it removes the rows which contain any of these outlier values (above or below the bounds)
    '''
    #start_size = df.shape[0]
    outlier_cutoffs = {}
    # Take out the features that I do not want to remove 'outliers' from: basically non-continuous features
    for col in df.drop(columns = cols_to_remove).columns:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = (q3 - q1)
        if q1-(k * iqr) < 0:
            low_outlier = 0
        else:
            low_outlier = q1 - (k * iqr)
        high_outlier = q3 + (k * iqr)

        outlier_cutoffs[col] = {}
        outlier_cutoffs[col]['upper_bound'] = high_outlier
        outlier_cutoffs[col]['lower_bound'] = low_outlier
    
    for col in df.drop(columns = cols_to_remove).columns:
        
        col_upper_bound = outlier_cutoffs[col]['upper_bound']
        col_lower_bound = outlier_cutoffs[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[df[col] <= col_upper_bound]
        df = df[df[col] >= col_lower_bound]
    #end_size = df.shape[0]
    #print(f'Removed {start_size - end_size} rows, or {(100*(start_size-end_size)/start_size):.2f}%')
    return df

def encoder(df,feature):
    '''
    Simple encoder of df - put in features and get dummies!
    '''
    dummy_df = pd.get_dummies(df[[feature]], dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def feature_adds(df):
    '''
    This adds features based on initial analysis and is SPECIFIC to the zillow clusering project
    Specifically, this adds a feature comparing structure vs land value, as well as a clustering feature
    '''
    df['structure_value_ratio'] = df.structurevalue/df.salesvalue

    kmeans = KMeans(n_clusters = 175)
    kmeans.fit(df[['latitude','longitude']])
    df['geo_cluster'] = kmeans.predict(df[['latitude','longitude']])
    geo_clusters = pd.DataFrame(df.groupby('geo_cluster').salesvalue.mean().sort_values()).rename(columns = {'salesvalue':'geo_sales_value'})
    
    # scaler = MinMaxScaler()
    # scaler.fit(geo_clusters)

    # # Fit the data
    # geo_clusters = pd.DataFrame(scaler.transform(geo_clusters),columns=geo_clusters.columns.values).set_index([geo_clusters.index.values])
    # geo_clusters['geo_cluster'] = geo_clusters.index

    df = df.merge(geo_clusters, how='inner', on='geo_cluster')
    df = df.drop(columns = ['geo_cluster'])  

    return df         

def scale_zillow(df_train,df_validate,df_test):
    '''
    Scales zillow dataframe using min-max scaler.
    Also creates dummy variables for categorical variable, location ('county')
    Must input split dataframe!
    '''
    # Create the object
    scaler = MinMaxScaler()
    scaler.fit(df_train.drop(columns=['abs_logerror','county','transactiondate','landusecode']))

    # Fit the data
    df_train_scaled = pd.DataFrame(scaler.transform(df_train.drop(columns=['abs_logerror','county','transactiondate','landusecode'])),columns=df_train.drop(columns=['abs_logerror','county','transactiondate','landusecode']).columns.values).set_index([df_train.index.values])
    df_validate_scaled = pd.DataFrame(scaler.transform(df_validate.drop(columns=['abs_logerror','county','transactiondate','landusecode'])),columns=df_validate.drop(columns=['abs_logerror','county','transactiondate','landusecode']).columns.values).set_index([df_validate.index.values])
    df_test_scaled = pd.DataFrame(scaler.transform(df_test.drop(columns=['abs_logerror','county','transactiondate','landusecode'])),columns=df_test.drop(columns=['abs_logerror','county','transactiondate','landusecode']).columns.values).set_index([df_test.index.values])

    df_train_scaled['abs_logerror'] = df_train['abs_logerror']
    df_validate_scaled['abs_logerror'] = df_validate['abs_logerror']
    df_test_scaled['abs_logerror'] = df_test['abs_logerror']

    return df_train_scaled, df_validate_scaled, df_test_scaled

def full_wrangle(df):
    '''
    This function combines all of the wrangling from above and explored in the workbook
    It returns all the necessary dataframes for EDA and modeling
    '''
    start_shape = df.shape
    # Removes dupliate transactions for a property (retaining last one)
    df = df.sort_values('transactiondate').drop_duplicates(subset='parcelid', keep = 'last')
    
    # Removes columns for null destroyer
    columns_to_remove = [
    'id.1', # different values but redundant with id
    'transactiondate.1', # same as transactiondate
    'logerror.1', # same as logerror
    'propertylandusetypeid', # same for all properties: 261
    'censustractandblock', # same as rawcensustractandblock, except 10**6 higher
    'finishedsquarefeet12', # same as calculatedfinsihedsquarefeet but with 247 less rows
    'regionidcounty', # same as fips - found in distribution analysis
    'roomcnt', # too many zeros - found in distribution analysis
    'assessmentyear', # All 2016
    'regionidzip', # Inaacurate (all zips in Oregon, not SoCal)
    'rawcensustractandblock', # unhelpful - discovered in distribution analysis
    'bathroomcnt', # redundant with calculatedbathnbr
    'fullbathcnt', # redundant with calculatedbathnbr
    'id', # not impactful
    'regionidcity', # not helpful, but used total quantity of these for geo clustering
    'parcelid'] # No longer needed

    df = data_prep(df,cols_to_remove = columns_to_remove,prop_required_column=.75, prop_required_row=.75)
    
    df = df.rename(columns={'calculatedbathnbr':'bathrooms','bedroomcnt':'bedrooms','calculatedfinishedsquarefeet':'sqft','fips':'county','lotsizesquarefeet':'lotsize','propertycountylandusecode':'landusecode','structuretaxvaluedollarcnt':'structurevalue','taxvaluedollarcnt':'salesvalue','landtaxvaluedollarcnt':'landvalue'})
    df['county'] = np.where(df.county == 6037, 'Los Angeles', np.where(df.county == 6059, 'Orange','Ventura'))
    df = df.astype({'sqft':'int', 'bedrooms':'int', 'lotsize':'int', 'yearbuilt':'int','structurevalue':'int','salesvalue':'int','landvalue':'int','taxamount':'int'})
    df['abs_logerror'] = abs(df.logerror)
    df = df.drop(columns = 'logerror')

    df.drop(labels=[4294,12465], axis=0,inplace = True)
    
    outlier_columns_to_remove = ['abs_logerror','transactiondate','county','latitude','landusecode','longitude']
    
    df = outlier_destroyer(df,2.5, cols_to_remove = outlier_columns_to_remove) 
    
    df = feature_adds(df)

    df = encoder(df,'county')
    df = encoder(df,'landusecode')

    end_shape = df.shape
    print(f'\n-----\nColumns reduced by {start_shape[1] - end_shape[1]} to {end_shape[1]}, and rows reduced by {start_shape[0] - end_shape[0]} to {end_shape[0]}')
    
    train, validate, test = splitter(df)

    X_train_exp = train.drop(columns = ['abs_logerror'])

    train_scaled, validate_scaled, test_scaled = scale_zillow(train,validate,test)

    X_train = train_scaled.drop(columns=['abs_logerror'])
    y_train = train_scaled.abs_logerror

    X_validate = validate_scaled.drop(columns=['abs_logerror'])
    y_validate = validate_scaled.abs_logerror

    X_test = test_scaled.drop(columns=['abs_logerror'])
    y_test = test_scaled.abs_logerror

    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test






