import pandas as pd
import os
from env import get_db_url


def get_zillow_data():
    '''
    Acquires zillow dataframe based on SQL query found below.  This SQL query pulls in all features
    from Properties 2017 that are Single Family Residential, along with their transaction data
    joined through parcelid
    '''
    filename = 'zillow_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql(
            '''
            SELECT DISTINCT
                *,
                pd.logerror,
                pd.transactiondate
            FROM
                properties_2017 AS p
                    RIGHT JOIN
                predictions_2017 AS pd USING (parcelid)
            WHERE
                propertylandusetypeid = 261;
            '''
            ,
            get_db_url('zillow')
        )

        df.to_csv(filename)

        return df