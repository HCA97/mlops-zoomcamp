def setup():
    import pandas as pd
    from datetime import datetime
    
    def dt(hour, minute, second=0): 
        return datetime(2022, 1, 1, hour, minute, second)


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

    options = {
        'client_kwargs': {
            'endpoint_url': "http://0.0.0.0:4566"
        }
    }
    input_file = 's3://nyc-duration/in/0000-00.parquet'


    # aws --endpoint-url http://0.0.0.0:4566 s3 ls s3://nyc-duration/in/
    # 2023-07-08 22:25:06       3667 00-0000.parquet
    df.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

setup()

import os

os.environ['OUTPUT_FILE_PATTERN'] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
os.environ['S3_ENDPOINT_URL'] = "http://0.0.0.0:4566"


from batch import main

main(0, 0)