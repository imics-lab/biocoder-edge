import yaml
import sys
import argparse
import psycopg2

# This script is in the 'scripts/' directory. We don't need to adjust the Python
# path because we are not importing from 'src/'. We just need to make sure
# the config path is correct relative to where the script is run.

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file from the project root.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure you are running this script from the project's root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def main():
    """
    Main function to test the PostgreSQL database connection.
    """
    parser = argparse.ArgumentParser(
        description="Test the PostgreSQL database connection using settings from config.yaml.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)."
    )
    args = parser.parse_args()

    print("--- Starting Database Connection Test ---")

    # 1. Load configuration and extract the database section
    config = load_config(args.config)
    try:
        db_config = config['uploader']['database']
    except KeyError:
        print("Error: 'uploader' or 'database' section not found in the configuration file.")
        sys.exit(1)

    print(f"Attempting to connect to database '{db_config.get('dbname')}' at {db_config.get('host')}:{db_config.get('port')}...")

    db_conn = None
    try:
        # 2. Attempt to establish the connection
        # The **db_config unpacks the dictionary into keyword arguments
        # that psycopg2.connect() expects (e.g., host='...', port='...', etc.)
        db_conn = psycopg2.connect(**db_config)
        
        # 3. If connection succeeds, run a simple query to verify
        cursor = db_conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        cursor.close()

        print("\n-------------------------------------------")
        print("✅ SUCCESS: Database connection established!")
        print(f"PostgreSQL Version: {db_version[0]}")
        print("-------------------------------------------")

    except psycopg2.Error as e:
        # 4. If connection fails, provide a detailed and helpful error message
        print("\n-------------------------------------------")
        print("❌ FAILURE: Could not connect to the database.")
        print(f"Error Details: {e}")
        print("\n--- Debugging Checklist ---")
        print("1. Is the database server running on the host machine?")
        print("2. Is the `host` IP address or hostname in config.yaml correct?")
        print("3. Is the `port` in config.yaml correct (usually 5432)?")
        print("4. Is there a firewall blocking the connection on the server or network?")
        print("5. Are the `dbname`, `user`, and `password` credentials correct?")
        print("6. Does the database user have permission to connect from this device's IP address? (Check PostgreSQL's pg_hba.conf file on the server).")
        print("-------------------------------------------")
        sys.exit(1)

    finally:
        # 5. Ensure the connection is always closed if it was opened
        if db_conn:
            db_conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()