
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_prepare_data():
    from batch import prepare_data

    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual = prepare_data(df, ['PULocationID', 'DOLocationID'])
    expected = pd.DataFrame([
                ('-1', '-1', dt(1, 2), dt(1, 10), 8.0),
                ('1', '-1', dt(1, 2), dt(1, 10), 8.0),
                ('1', '2', dt(2, 2), dt(2, 3), 1.0), 
            ], columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])


    assert len(actual) == 3
    for col in actual.columns:
        a = list(actual[col])
        e = list(expected[col])
        assert a == e