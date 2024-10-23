import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

nltk.download('stopwords')

conn = psycopg2.connect(
    host="h-pgsql01.pgj.rj.gov.br",
    database="gate",
    user="gate",
    password="gatehom2020",
    port="5432"  # Default PostgreSQL port
)

conn.autocommit = True

cursor = conn.cursor()

query = '''
SELECT
  "ITCN_DK" "DOCUMENTO"
, "TEXTO"
--, 'IT' "ORIGEM"
FROM "stage"."PRODATA_IT_TEXTO"
--union
--SELECT
--  'DESP'||to_char("ID_DOCUMENTO", 'fm000000000000')  "DOCUMENTO"
--, "DESPACHO" "TEXTO"
--, 'DESPACHO' "ORIGEM"
--FROM "stage"."SEI_SAT_DSPCH"
--union
--SELECT
--  'SAT'||to_char("ID_DOCUMENTO", 'fm000000000000')  "DOCUMENTO"
--, "DUVIDA" "TEXTO"
--, 'SAT' "ORIGEM"
--FROM "stage"."SEI_SATs"
--where "DUVIDA" is not null
--  and (btrim("DUVIDA", ' ') <> '' or btrim("DUVIDA", ' ') is not null)
--limit 100;
'''

dados = pd.read_sql(query, conn)

# Vetorização dos textos usando TF-IDF
for i, reg in dados.iterrows():
    dados.at[i,'TEXTO'] = re.sub(r'\W', ' ', reg['TEXTO'])
    dados.at[i,'TEXTO'] = re.sub(r'\s+[a-zA-Z]\s+', ' ', reg['TEXTO'])
    dados.at[i,'TEXTO'] = re.sub(r'\^[a-zA-Z]\s+', ' ', reg['TEXTO'])
    dados.at[i,'TEXTO'] = re.sub(r'\s+', ' ', reg['TEXTO'], flags=re.I)
    dados.at[i,'TEXTO'] = re.sub(r'^b\s+', '', reg['TEXTO'])
    dados.at[i,'TEXTO'] = re.sub(r'^(.*)SOLICITANTE', '', reg['TEXTO'], flags=re.I)
    dados.at[i,'TEXTO'] = reg['TEXTO'].lower()

tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
tfidf_matrix = tfidf_vectorizer.fit_transform(dados['TEXTO'])

# Calcular a similaridade do cosseno
cosine_sim = cosine_similarity(tfidf_matrix)

# Exibir a matriz de similaridade (opcional)
#similarity_df = pd.DataFrame(cosine_sim, columns=dados['DOCUMENTO'], index=dados['DOCUMENTO'])

s = []

for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.5 and i != j:  # Limite de similaridade (ajustável)
            s.append({'IT1': dados['DOCUMENTO'][i]
                     , 'IT2': dados['DOCUMENTO'][j]
                     , 'GS': cosine_sim[i, j]})
#            print(f"Alta similaridade entre '{dados['DOCUMENTO'][i]}' e '{dados['DOCUMENTO'][j]}': {cosine_sim[i, j]:.2f}")

#pd.DataFrame(s).to_csv('F:\\temp\\similarity_df.csv', index=False)

# Clustering dos documentos com base na similaridade (por exemplo, aglomerativo)
#clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.3)
#labels = clustering_model.fit_predict(1 - cosine_sim)

# Adicionar os rótulos de grupo ao DataFrame original
#dados['GRUPO'] = labels

# Exibir os documentos agrupados
#print(dados[['DOCUMENTO', 'ORIGEM', 'GRUPO']])

###### LIMPA stage."IT_SIMILAR" e insere as previsões

truncate_query = 'truncate table stage."IT_SIMILAR"'

# Execute the command
cursor.execute(truncate_query)

# Insert DataFrame rows one by one
for i,row in pd.DataFrame(s).iterrows():
    qins = ('''
            INSERT INTO stage."IT_SIMILAR"
            ("ITCN_DK_1", "ITCN_DK_2", "GS")
            VALUES(%i, %i, %f);
        '''
            % (row['IT1']
           , row['IT2']
           , row['GS']
        ))
    cursor.execute(qins)

# Close the cursor and connection
cursor.close()
conn.close()
