import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering: ratios, encoding, drop collinears

    Args:
        df: cleaned DataFrame
    
    Returns:
        Feature-engineered DataFrame
    """
    
    df = df.copy()
    # combine features into more meaningful ones
    df['dasher_availability_ratio'] = df['total_busy_dashers'] / (df['total_onshift_dashers']+1)
    df['non_prep_duration'] = df['estimated_order_place_duration'] + df['estimated_store_to_consumer_driving_duration']
    df['price_range'] = df['max_item_price'] - df['min_item_price']
    df['avg_item_price'] = df['subtotal'] / df['total_items']
    df['distinct_items_ratio'] = df['num_distinct_items'] / df['total_items']

    # drop redundant features
    collinear_to_drop = [
        'total_onshift_dashers',
        'total_busy_dashers',
        'subtotal',
        'num_distinct_items',
        'min_item_price',
        'max_item_price',
        'non_prep_duration'
    ]

    df.drop(columns=collinear_to_drop, inplace=True)

    # delivery_duration has one extreme value so remove it
    df = df[df['delivery_duration'] != 332482.0]

    # transform all the numeric_cols except the target (delivery_duration)
    to_logtransform = df.drop(columns=[
                                'market_id',
                                'store_id',
                                'store_primary_category',
                                'order_protocol',
                                'nan_store_primary_category',
                                'delivery_duration'
                                ])
    # log transform to all numeric cols
    for col in to_logtransform:
        df[col] = np.log1p(df[col])
    
    # convert -inf and inf values in price_range to np.nan
    df.replace([-np.inf, np.inf], np.nan, inplace=True)

    # -- ENCODING --
    to_encode = ['market_id', 'nan_store_primary_category', 'order_protocol']
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    encoded = encoder.fit_transform(df[to_encode])
    encoded_cols = encoder.get_feature_names_out(to_encode)
    
    # rename columns for custom prefixes
    encoded_cols = [
        col.replace('nan_store_primary_category_', 'category_')
           .replace('order_protocol_', 'protocol_')
        for col in encoded_cols
    ]

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=to_encode), encoded_df], axis=1)

    df.drop(columns=['store_id', 'store_primary_category'], inplace=True)

    # drop all na values
    df.dropna(inplace=True)

    return df