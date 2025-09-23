import datetime

import pandas as pd
import polars as pl
import sqlalchemy

from sqlalchemy import Engine
from util.postgres import query, query_polars


def query_nodes_author(conn: sqlalchemy.engine.base.Connection,
                       table_name: str) -> pd.DataFrame:
    # Get all authors data and value metrics about their collaboration
    author_query: str = f"""
    SELECT author_id,
           publication_count,
           article_citation_normalized_count,
           collaboration_novelty_index,
           author_embedding::FLOAT8[]               AS author_embedding,
           keyword_popularity_embedding::FLOAT8[]   AS keyword_popularity_embedding 
    FROM {table_name};
    """
    print("Querying author nodes...")
    author_df: pd.DataFrame = query(conn=conn, query_str=author_query)

    return author_df


def query_author_keyword_embeddings(conn: sqlalchemy.engine.base.Connection,
                                    filter_dt: datetime.datetime) -> pd.DataFrame:
    # Get all author keyword embeddings
    author_query: str = f"""
    WITH author_top_keywords AS (SELECT c.author_id,
                                        t.article_keyword,
                                        ARRAY [ COALESCE(
                                                AVG(CASE WHEN t.kpi_year = - 1 THEN t.keyword_popularity_index END), 0)
                                            , COALESCE(AVG(CASE WHEN t.kpi_year = - 2 THEN t.keyword_popularity_index END),
                                                       0)
                                            , COALESCE(AVG(CASE WHEN t.kpi_year = - 3 THEN t.keyword_popularity_index END),
                                                       0)
                                            , COALESCE(AVG(CASE WHEN t.kpi_year = - 4 THEN t.keyword_popularity_index END),
                                                       0)
                                            , COALESCE(AVG(CASE WHEN t.kpi_year = - 5 THEN t.keyword_popularity_index END),
                                                       0) ]
                                            AS keyword_popularity_index_arr
                                 FROM g_eucohm_author_top_200_keywords t
                                          INNER JOIN fct_collaboration c
                                                     ON c.article_id = t.article_id
                                 WHERE -- To prevent data leakage, we only consider articles published before the given date
                                       t.article_publication_dt <= '{filter_dt}'
                                 GROUP BY c.author_id, t.article_keyword),
         top_keywords AS (SELECT DISTINCT article_keyword
                          FROM g_eucohm_author_top_200_keywords),
         author_keywords AS (SELECT a.author_id,
                                    tk.article_keyword,
                                    COALESCE(atk.keyword_popularity_index_arr, ARRAY [0,0,0,0,0]) AS keyword_popularity_index_arr
                             FROM g_included_author a
                                      CROSS JOIN top_keywords tk
                                      LEFT JOIN author_top_keywords atk
                                                ON atk.author_id = a.author_id
                                                    AND tk.article_keyword = atk.article_keyword)
    SELECT ak.author_id
         -- vector containing all the keyword popularity indices for the given author ordered by keyword
         , ARRAY_AGG(k.keyword_popularity_index_value
                     ORDER BY ak.article_keyword, k.keyword_popularity_index_row_number) keyword_popularity_embedding
    FROM author_keywords AS ak,
         LATERAL UNNEST(keyword_popularity_index_arr) WITH ORDINALITY AS k(keyword_popularity_index_value, keyword_popularity_index_row_number)
    GROUP BY author_id;
    """
    print("Querying author keyword popularity embeddings...")
    author_df: pd.DataFrame = query(conn=conn, query_str=author_query)

    return author_df


def query_article_embeddings(engine: Engine,
                             filter_dt: datetime.datetime,
                             batch_size: int = 10000) -> pl.DataFrame:
    with engine.raw_connection().cursor() as cur:
        cur.execute(f"""
            SELECT article_id, article_publication_dt, article_embedding::FLOAT8[]
            FROM g_included_article_embedding
            -- To prevent data leakage we only learn from articles published before cutoff date
            WHERE article_publication_dt <= '{filter_dt}'
        """)
        # Initialize the Polars DataFrame
        article_embedding_df: pl.DataFrame = pl.DataFrame()
        ix = 0
        # Fetch in chunks
        while True:
            rows = cur.fetchmany(size=batch_size)
            if not rows:
                break
            # Append the rows to the Polars DataFrame
            df_chunk = pl.DataFrame(rows, schema=["article_id", "article_publication_dt", "article_embedding"],
                                    orient="row")

            # Concatenate chunk with the master DataFrame
            article_embedding_df = pl.concat([article_embedding_df, df_chunk], how="vertical")
            print(f"Rows fetched {batch_size} for batch {ix}")
            ix += 1

        return article_embedding_df


def query_nodes_author_articles(conn: sqlalchemy.engine.base.Connection, filter_dt: datetime.datetime) -> pl.DataFrame:
    # Get all authors data and value metrics about their collaboration
    author_articles_query: str = f"""
        SELECT author_id,
               article_id,
               article_publication_dt,
               article_citation_normalized_count,
               collaboration_novelty_index
        FROM g_eucohm_node_author_article
        -- To prevent data leakage we only learn from articles published before cutoff date
        WHERE article_publication_dt <= '{filter_dt}'
        """
    print("Querying author articles...")
    author_articles_df: pl.DataFrame = query_polars(conn=conn, query_str=author_articles_query)

    return author_articles_df


def query_edges_co_authors(conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
    # Get all edges between authors and co-authors
    coauthored_query = f"""
    SELECT author_id,
           co_author_id,
           time,
           1 + eutopia_collaboration_count AS weight
    FROM g_eucohm_edge_co_authors
    """
    print("Querying co-authorship edge data...")
    coauthored_df = query(conn=conn, query_str=coauthored_query)

    return coauthored_df


def query_nth_time_percentile(conn: sqlalchemy.engine.base.Connection, percentile: float = 0.8) -> datetime.datetime:
    # Get all edges between authors and co-authors
    nth_time_percentile_query = f"""
        -- To prevent data leakage we only learn from articles published before cutoff date
        SELECT TO_DATE(CAST(PERCENTILE_CONT({percentile}) WITHIN GROUP ( ORDER BY CAST(time AS INT)) AS TEXT), 'YYYYMMDD') AS time
        FROM g_eucohm_edge_co_authors
    """
    print("Querying n-th time percentile...")
    nth_time_percentile_df = query(conn=conn, query_str=nth_time_percentile_query)

    return nth_time_percentile_df['time'][0]
