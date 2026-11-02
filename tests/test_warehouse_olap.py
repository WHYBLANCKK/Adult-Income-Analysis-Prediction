"""
test_warehouse_olap.py - Validates SQLite Star Schema and 4 core OLAP operations.
"""

import sqlite3
import re
import pytest
import pandas as pd
from pathlib import Path

def test_star_schema_tables_and_queries():
    db_path = Path('warehouse/adult_dw.db')
    assert db_path.exists(), "warehouse/adult_dw.db does not exist"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check table existence
    tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    for expected_table in ['FactIncome', 'DimDemographics', 'DimEducation', 'DimOccupation', 'DimGeography']:
        assert expected_table in tables, f"Missing table {expected_table} in warehouse"
    
    # Check record counts
    fact_count = cursor.execute("SELECT COUNT(*) FROM FactIncome;").fetchone()[0]
    assert fact_count == 32561, f"Expected 32,561 rows in FactIncome, got {fact_count}"
    
    # Check 4 OLAP Queries
    with open('warehouse/olap_queries.sql', 'r') as f:
        clean_sql = re.sub(r'--.*', '', f.read())
    queries = [q.strip() for q in clean_sql.split(';') if q.strip()]
    
    assert len(queries) == 4, f"Expected exactly 4 OLAP queries, found {len(queries)}"
    
    # Query 1: Roll-up
    res1 = pd.read_sql_query(queries[0], conn)
    assert len(res1) > 0, "Query 1 (Roll-up) returned 0 rows"
    assert 'workclass' in res1.columns and 'occupation' in res1.columns
    
    # Query 2: Drill-down
    res2 = pd.read_sql_query(queries[1], conn)
    assert len(res2) > 0, "Query 2 (Drill-down) returned 0 rows"
    assert 'education_level' in res2.columns and 'marital_status' in res2.columns
    
    # Query 3: Slice
    res3 = pd.read_sql_query(queries[2], conn)
    assert len(res3) > 0, "Query 3 (Slice) returned 0 rows"
    assert 'female_high_income_pct' in res3.columns
    
    # Query 4: Dice
    res4 = pd.read_sql_query(queries[3], conn)
    assert len(res4) > 0, "Query 4 (Dice) returned 0 rows"
    assert 'age_band' in res4.columns and 'high_income_pct' in res4.columns
    
    conn.close()
