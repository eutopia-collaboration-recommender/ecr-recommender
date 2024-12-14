import pandas as pd
import sqlalchemy
from box import Box

# Job configuration
job_config = {
    'source_json_filename': 'GRAPH_HOMOGENEOUS_NODE_AUTHOR'
}

if __name__ == '__main__':
    # Read settings from config file
    config = Box.from_yaml(filename="../config.yaml")

    # Connect to Postgres
    engine = sqlalchemy.create_engine(
        f"postgresql://{config.POSTGRES.USERNAME}:{config.POSTGRES.PASSWORD}@{config.POSTGRES.HOST}:{config.POSTGRES.PORT}/{config.POSTGRES.DATABASE}"
    )
    connection = engine.connect()

    # Read JSON file
    df = pd.read_json(f"../data/{job_config['source_json_filename']}.json", lines=True)
    df.columns = df.columns.str.lower()

    # Write the DataFrame to the PostgreSQL database
    df.to_sql(
        job_config['source_json_filename'].lower(),
        con=connection,
        schema=config.POSTGRES.BQ_SCHEMA,
        if_exists='replace',
        index=False
    )
