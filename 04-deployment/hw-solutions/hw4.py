import argparse


import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np

CATEGORICAL = ['PULocationID', 'DOLocationID']

def load_models(model_path: str = 'model.bin') -> tuple:
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


def read_data(filename : str) -> pd.DataFrame:
    global CATEGORICAL
    print(f'Dowloading {filename}...')

    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df: pd.DataFrame, dv: DictVectorizer, model: LinearRegression) -> np.ndarray:
    global CATEGORICAL
    print(f'Doing prediction for {len(df)} rows...')
    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred


def main(taxi_type: str, month: int, year: int, output_dir: str):
    data_ulr = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(data_ulr)
    dv, model = load_models()
    pred = predict(df, dv, model)
    print(f'mean pred: {pred.mean()}, std pred: {pred.std()}...')

    df['predictions'] = pred
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    output_path = f'{output_dir}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    print(f'Saving the result to {output_path}...')
    df_results = df[['ride_id', 'predictions']]
    df_results.to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxi-type', default='yellow', choices=['yellow', 'green'])
    parser.add_argument('--month', type=int, default=1, choices=list(range(1, 13)))
    parser.add_argument('--year', type=int, default=2022)
    parser.add_argument('--output-path', default='.')

    args = parser.parse_args()
    main(args.taxi_type, args.month, args.year, args.output_path)
    
    


