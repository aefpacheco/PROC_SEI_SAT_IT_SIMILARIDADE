import psycopg2
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from psycopg2.extras import execute_values

nltk.download('stopwords')

conn = psycopg2.connect(
    host="h-pgsql01.pgj.rj.gov.br",
    database="gate",
    user="gate",
    password="gatehom2020",
    port="5432"
)

conn.autocommit = True

cursor = conn.cursor()

query = '''
SELECT
  "ITCN_DK" "DOCUMENTO"
, "TEXTO"
FROM "stage"."PRODATA_IT_TEXTO"
--limit 100;
'''

dados = pd.read_sql(query, conn)

# Limpeza dos textos usando expressões regulares
dados['TEXTO'] = dados['TEXTO'].str.lower()
dados['TEXTO'] = dados['TEXTO'].str.replace(r'\W+', ' ', regex=True, flags=re.I)
dados['TEXTO'] = dados['TEXTO'].str.replace(r'\s+', ' ', regex=True, flags=re.I)
dados['TEXTO'] = dados['TEXTO'].str.replace(r'\s+[a-z]\s+', ' ', regex=True, flags=re.I)
dados['TEXTO'] = dados['TEXTO'].str.replace(r'^[a-z0-9]\s+', '', regex=True, flags=re.I)
dados['TEXTO'] = dados['TEXTO'].str.replace(r'^(.*)SOLICITANTE', '', regex=True, flags=re.I)

tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
tfidf_matrix = tfidf_vectorizer.fit_transform(dados['TEXTO'])

# Aplicar o KMeans para clusterização dos documentos
num_clusters = 12  # Defina o número de clusters desejado
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
dados['CLUSTER'] = kmeans_model.fit_predict(tfidf_matrix)

# Calcular a similaridade do cosseno
cosine_sim = cosine_similarity(tfidf_matrix)

s = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.5 and i != j:
            s.append({'IT1': dados['DOCUMENTO'][i]
                         , 'IT2': dados['DOCUMENTO'][j]
                         , 'GS': cosine_sim[i, j]
                         , 'CLUSTER1': dados['CLUSTER'][i]
                         , 'CLUSTER2': dados['CLUSTER'][j]
                      })

cursor.execute('truncate table stage."IT_SIMILAR"')

# Inserção em lote dos resultados
insert_query = '''
    INSERT INTO stage."IT_SIMILAR" ("ITCN_DK_1", "ITCN_DK_2", "GS", "CLUSTER1", "CLUSTER2")
    VALUES %s
'''
values = [(float(row['IT1']), float(row['IT2']), float(row['GS']), int(row['CLUSTER1']), int(row['CLUSTER2'])) for row in s]
execute_values(cursor, insert_query, values)

# Close the cursor and connection
cursor.close()
conn.close()