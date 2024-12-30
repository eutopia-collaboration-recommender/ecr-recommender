from io import StringIO

import pandas as pd
import polars as pl
import psycopg2
import sqlalchemy.engine.base
from sqlalchemy import create_engine, Engine


def create_connection(username: str,
                      password: str,
                      host: str,
                      port: str,
                      database: str,
                      schema: str) -> psycopg2.extensions.connection:
    """
    Create a connection to Postgres
    :param username: Postgres username
    :param password: Postgres password
    :param host: Postgres host
    :param port: Postgres port
    :param database: Postgres database
    :param schema: Postgres schema
    :return: Postgres connection
    """
    # Define the connection string
    conn_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    # Create the connection
    conn = psycopg2.connect(conn_string)
    # Set the schema
    conn.cursor().execute(f'SET search_path TO {schema}')
    # Return the connection
    return conn


def create_sqlalchemy_engine(username: str,
                             password: str,
                             host: str,
                             port: str,
                             database: str,
                             schema: str) -> Engine:
    """
    Create a connection to Postgres using SQLAlchemy
    :param username: Postgres username
    :param password: Postgres password
    :param host: Postgres host
    :param port: Postgres port
    :param database: Postgres database
    :param schema: Postgres schema
    :return: SQLAlchemy connection
    """
    # Define the connection string
    conn_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    # Create the connection
    engine = create_engine(
        conn_string,
        connect_args={'options': '-csearch_path={}'.format(schema)})
    # Return the connection
    return engine


def write_table(conn: psycopg2.extensions.connection,
                table_name: str,
                df: pd.DataFrame,
                merge_on: list = None,
                mode: str = 'overwrite') -> int:
    """
    Write data to a Postgres table (replace the table if it already exists).
    If `merge_on` is provided, performs a MERGE (upsert); otherwise, appends rows.
    :param mode:  'overwrite' or 'append'
    :param conn Postgres connection
    :param table_name: Postgres table name
    :param df: Pandas DataFrame with the data
    :param merge_on: List of columns to merge on
    :return: Number of rows written

    """

    # Write the DataFrame to a string buffer
    string_io = StringIO()
    df.to_csv(string_io, index=False, header=False)
    string_io.seek(0)

    # Generate column names
    columns = ', '.join(df.columns)
    if mode == 'overwrite':
        with conn.cursor() as cursor:
            # Create table and populate using `COPY`
            temp_table_name = f"{table_name}_temp"
            cursor.copy_expert(
                sql=f"COPY {temp_table_name} ({columns}) FROM STDIN WITH CSV",
                file=string_io
            )
            cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_table_name}")
            cursor.execute(f"DROP TABLE {temp_table_name}")



    else:

        with conn.cursor() as cursor:
            if merge_on:
                # MERGE Logic
                excluded_columns = ['row_created_at']
                update_clause = ', '.join([f"{col} = source.{col}" for col in df.columns if
                                           col not in merge_on and col not in excluded_columns])
                insert_columns = ', '.join(df.columns)
                insert_values = ', '.join([f"source.{col}" for col in df.columns])
                on_clause = ' AND '.join([f"source.{col} = target.{col}" for col in merge_on])

                merge_query = f"""
                    MERGE INTO {table_name} AS target
                    USING (SELECT * FROM temp_table) AS source
                    ON {on_clause}
                    WHEN MATCHED THEN
                      UPDATE SET {update_clause}
                    WHEN NOT MATCHED THEN
                      INSERT ({insert_columns})
                      VALUES ({insert_values});
                """

                # Create a temporary table to hold the data
                temp_table = f"{table_name}_temp"
                cursor.execute(f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING ALL)")

                # Load the data into the temporary table
                cursor.copy_expert(
                    sql=f"COPY {temp_table} ({columns}) FROM STDIN WITH CSV",
                    file=string_io
                )

                # Execute the merge query
                cursor.execute(merge_query.replace("temp_table", temp_table))
            else:
                # APPEND Logic
                cursor.copy_expert(
                    sql=f"COPY {table_name} ({columns}) FROM STDIN WITH CSV",
                    file=string_io
                )

        # Commit the transaction
        conn.commit()

    # Return the DataFrame
    return len(df)


def query(conn: sqlalchemy.engine.base.Connection, query_str: str) -> pd.DataFrame:
    """
    Query Postgres.
    :param conn: Postgres connection
    :param query: SQL query
    :return: Pandas DataFrame with the data
    """
    # Fetch the data
    df = pd.read_sql(query_str, conn)
    # Return the DataFrame
    return df


def query_polars(conn: sqlalchemy.engine.base.Connection, query_str: str) -> pl.DataFrame:
    """
    Query Postgres.
    :param conn: Postgres connection
    :param query: SQL query
    :return: Pandas DataFrame with the data
    """
    # Fetch the data
    df = pl.read_database(query_str, conn)
    # Return the DataFrame
    return df


def use_schema(conn: psycopg2.extensions.connection, schema: str) -> None:
    """
    Set the schema for the connection
    :param conn: Postgres connection
    :param schema: Postgres schema
    """
    # Set the schema
    conn.cursor().execute(f'SET search_path TO {schema}')
    # Commit the transaction
    conn.commit()
