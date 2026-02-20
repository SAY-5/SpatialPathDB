"""
Database connection utilities for PostgreSQL/PostGIS.
"""

import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    """Build database URL from environment variables or use default."""
    return os.getenv(
        'DATABASE_URL',
        'postgresql://pathdb_user:pathdb_pass@localhost:5432/spatialpathdb'
    )


def get_db_engine() -> Engine:
    """Create SQLAlchemy engine for ORM operations."""
    return create_engine(
        get_database_url(),
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )


@contextmanager
def get_db_connection(dict_cursor: bool = False) -> Generator:
    """
    Context manager for database connections.
    Automatically handles connection lifecycle and commits.
    """
    url = get_database_url()

    # Parse URL for psycopg2
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', '')

    # Extract components: user:pass@host:port/db
    auth, rest = url.split('@')
    user, password = auth.split(':')
    host_port, dbname = rest.split('/')

    if ':' in host_port:
        host, port = host_port.split(':')
    else:
        host = host_port
        port = '5432'

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        cursor_factory=RealDictCursor if dict_cursor else None
    )

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_copy(conn, table_name: str, columns: list, data_file: str) -> int:
    """
    Execute PostgreSQL COPY command for bulk data loading.
    Returns number of rows copied.
    """
    with conn.cursor() as cur:
        columns_str = ', '.join(columns)
        with open(data_file, 'r') as f:
            cur.copy_expert(
                f"COPY {table_name} ({columns_str}) FROM STDIN WITH CSV HEADER",
                f
            )
        return cur.rowcount
