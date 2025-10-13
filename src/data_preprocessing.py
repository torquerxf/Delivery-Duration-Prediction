import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean delivery dataset: create target column, handle missing values, invalid entries

    Args:
        df: Raw input DataFrame
    
    Returns:
        Cleaned DataFrame
    """

    df = df.copy()

    # convert the date features to appropriate dtype i.e dateTime
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

    # create target column delivery duration
    df['delivery_duration'] = (
        df['actual_delivery_time'] - df['created_at']
    ).dt.total_seconds()

    # ensure all entries are positive
    df = df[df['delivery_duration'] > 0]

    # drop the date columns
    df.drop(columns=['created_at', 'actual_delivery_time'], inplace=True)

    # contain all numeric features
    numeric_cols = ['total_items', 'subtotal', 'num_distinct_items', 'min_item_price', 'max_item_price', 'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_order_place_duration', 'estimated_store_to_consumer_driving_duration']

    # handle negative entries in numeric_cols; here they can't be negative
    for col in numeric_cols:
        df = df[df[col] > 0]

    # handle null values in store_primary_category with imputation
    store_id_list = df['store_id'].unique()
    store_primary_category_map = {store_id: df[df['store_id'] == store_id]['store_primary_category'].mode() for store_id in store_id_list}

    def fill(store_id):
        try:
            return store_primary_category_map[store_id][0]
        except:
            return np.nan
    
    df['nan_store_primary_category'] = df['store_id'].apply(fill)

    # replace any infinity entry to nan
    df.replace([-np.inf, np.inf], np.nan, inplace=True)

    return df

