from application import app

app.secret_key = "bigtuing"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'bigProject'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'