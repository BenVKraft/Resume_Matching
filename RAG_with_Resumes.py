import os
import json
import pyodbc
import struct
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from prettytable import PrettyTable
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# --- DB Connection Helper ---
def get_mssql_connection():
    entra_connection_string = os.getenv('ENTRA_CONNECTION_STRING')
    sql_connection_string = os.getenv('SQL_CONNECTION_STRING')
    if entra_connection_string:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        token = credential.get_token('https://database.windows.net/.default')
        token_bytes = token.token.encode('UTF-16LE')
        token_struct = struct.pack(f'<I{len(token_bytes)}s', len(token_bytes), token_bytes)
        SQL_COPT_SS_ACCESS_TOKEN = 1256
        conn = pyodbc.connect(entra_connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})
    elif sql_connection_string:
        conn = pyodbc.connect(sql_connection_string)
    else:
        raise ValueError("No valid connection string found in the environment variables.")
    return conn

# --- Vector Similarity Search in Azure SQL DB using VECTOR_DISTANCE ---
def vector_search_sql(query, num_results=5):
    load_dotenv()
    conn = get_mssql_connection()
    cursor = conn.cursor()
    user_query_embedding = get_embedding(query)
    sql_similarity_search = """
    SELECT TOP(?) filename, chunkid, chunk,
           1-vector_distance('cosine', CAST(CAST(? as NVARCHAR(MAX)) AS VECTOR(1536)), embedding) AS similarity_score,
           vector_distance('cosine', CAST(CAST(? as NVARCHAR(MAX)) AS VECTOR(1536)), embedding) AS distance_score
    FROM dbo.resumedocs
    ORDER BY distance_score 
    """
    cursor.execute(sql_similarity_search, num_results, json.dumps(user_query_embedding), json.dumps(user_query_embedding))
    results = cursor.fetchall()
    conn.close()
    return results

# --- Embedding Generation for Queries ---
import requests
def get_embedding(text):
    openai_embedding_model = os.getenv("AZOPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME")
    openai_url = os.getenv("AZOPENAI_ENDPOINT") + "openai/deployments/" + openai_embedding_model + "/embeddings?api-version=2023-05-15"
    openai_key = os.getenv("AZOPENAI_API_KEY")
    response = requests.post(openai_url,
        headers={"api-key": openai_key, "Content-Type": "application/json"},
        json={"input": [text]}
    )
    if response.status_code == 200:
        response_json = response.json()
        embedding = json.loads(str(response_json['data'][0]['embedding']))
        return embedding
    else:
        return None

# --- Azure OpenAI Client Setup ---
api_key = os.getenv("AZOPENAI_API_KEY")
azure_endpoint = os.getenv("AZOPENAI_ENDPOINT")
chat_model = 'gpt-4o'

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version="2024-12-01-preview"
)

# --- LLM Completion using Search Results ---
def generate_completion(search_results, user_input):
    system_prompt = '''
You are an intelligent & funny assistant who will exclusively answer based on the data provided in the `search_results`:
- Use the information from `search_results` to generate your top 3 responses. If the data is not a perfect match for the user's query, use your best judgment to provide helpful suggestions and include the following format:
  File: {filename}
  Chunk ID: {chunkid}
  Similarity Score: {similarity_score}
  Add a small snippet from the Relevant Text: {chunktext}
  Do not use the entire chunk
- Avoid any other external data sources.
- Add a summary about why the candidate maybe a goodfit even if exact skills and the role being hired for are not matching , at the end of the recommendations. Ensure you call out which skills match the description and which ones are missing. If the candidate doesnt have prior experience for the hiring role which we may need to pay extra attention to during the interview process.
'''
    messages = [{"role": "system", "content": system_prompt}]
    result_list = []
    for result in search_results:
        filename = result[0]
        chunkid = result[1]
        chunktext = result[2]
        similarity_score = result[3]
        result_list.append({
            "filename": filename,
            "chunkid": chunkid,
            "chunktext": chunktext,
            "similarity_score": similarity_score
        })
    messages.append({"role": "system", "content": f"{result_list}"})
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(model=chat_model, messages=messages, temperature=0)
    return response.dict()