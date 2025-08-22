import psycopg2

conn = psycopg2.connect(
    dbname="postgres",   # nombre de tu base de datos, por defecto es 'postgres'
    user="postgres",     # usuario, por defecto es 'postgres'
    password="186968",  # pon aquí la que configures
    host="localhost",
    port="5432"          # puerto por defecto
)

cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())

cur.close()
conn.close()
