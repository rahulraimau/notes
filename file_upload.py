import pandas as pd
import sqlite3

# Load CSV file
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Load Excel file
def load_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Excel loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return None

# Load JSON file
def load_json(file_path):
    try:
        df = pd.read_json(file_path)
        print("JSON loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

# Load data from SQL database
def load_sql(database, query):
    try:
        conn = sqlite3.connect(database)
        df = pd.read_sql_query(query, conn)
        conn.close()
        print("SQL data loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading SQL data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    csv_file = "data.csv"
    excel_file = "data.xlsx"
    json_file = "data.json"
    db_file = "database.db"
    sql_query = "SELECT * FROM table_name"

    df_csv = load_csv(csv_file)
    df_excel = load_excel(excel_file)
    df_json = load_json(json_file)
    df_sql = load_sql(db_file, sql_query)