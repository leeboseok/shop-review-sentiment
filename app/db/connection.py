import pymysql

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "boseok",
    "password": "iotiot",
    "database": "sentimentreview",
    "port": 3306
}

def get_db_connection():
    try:
        conn = pymysql.connect(
            **DB_CONFIG,
            cursorclass=pymysql.cursors.DictCursor,
            charset='utf8mb4'
        )
        return conn
    except pymysql.MySQLError as err:
        print(f"MySQL 연결 오류: {err}")
        return None
