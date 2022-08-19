import os
import json
import pandas as pd
from hdbcli import dbapi
from datetime import datetime
from config import column_keys

db_user = os.getenv("DB_USER", "")
db_password = os.getenv("DB_PSSW", "")
db_host = os.getenv("DB_HOST", "")
db_port = os.getenv("DB_PORT", "")
db_schema = os.getenv("DB_SCHEMA","")

conn = dbapi.connect(address=db_host, port=int(db_port), user=db_user, password=db_password)
cursor = conn.cursor()

def write_res_to_db(df: pd.DataFrame, algo_name: str, metrics: list = ['personalization','diversity','coverage']):
    table = 'ALGO_RECOMMENDATIONS'
    today = datetime.today().strftime('%Y-%m-%d')
    insert_query = f'''insert into "{db_schema}"."{table}" values ('%s','%s','%s','%s','%s','%s','%s')'''
    user_col = column_keys.user_col
    cursor.execute(f'''delete from "{db_schema}"."{table}" where "ALGO_NAME"='{algo_name}' ''')
    for i in df.index:
        score_dict = {m: round(df[m][i], 4) for m in metrics}
        cursor.execute(insert_query%(today, algo_name, df[user_col][i].replace("'",''), json.dumps(score_dict).replace("'",''), json.dumps(df['url'][i]).replace("'",''), json.dumps(df[column_keys.item_name_col][i]).replace("'",'')))
    print(f'Finished writing {algo_name} recommendation results to DB')



