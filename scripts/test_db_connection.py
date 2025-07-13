#!/usr/bin/env python3
"""
scripts/test_db_connection.py

A simple script to verify connectivity to the PostgreSQL database
defined in config/config.yaml. Prints success or error details.
"""
import psycopg2
import yaml
import sys


def load_config(path="config/config.yaml"):
    """
    Loads the YAML configuration file from the project root.
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def test_db_connection(config):
    """
    Attempts to connect to the PostgreSQL database using the provided
    configuration. Exits with code 0 on success, 1 on failure.
    """
    db_cfg = config['uploader']['database']
    host = db_cfg.get('host')
    port = db_cfg.get('port')
    dbname = db_cfg.get('dbname')
    user = db_cfg.get('user')
    print(f"🔌 Testing database connection to {host}:{port}/{dbname} as user '{user}'...")
    try:
        # Use a short timeout to fail fast if server is unreachable
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=db_cfg.get('password'),
            connect_timeout=5
        )
        conn.close()
        print("✅ Successfully connected to PostgreSQL database.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Allow overriding the config path via command-line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    config = load_config(config_path)
    test_db_connection(config)
