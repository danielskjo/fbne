import mysql.connector

db = mysql.connector.connect(
    host='',
    user='',
    password='',
    database='fbne'
)

cursor = db.cursor()