# Databricks notebook source
# MAGIC %md
# MAGIC # Speedy Motors Customer Preference RAG App
# MAGIC ### Retrieval Augmented Generation (RAG) with Salesforce Data
# MAGIC

# COMMAND ----------

# MAGIC %pip install --force-reinstall databricks-feature-store 
# MAGIC %pip install --force-reinstall databricks_vectorsearch 
# MAGIC %pip install --force-reinstall -v langchain openai
# MAGIC %pip install databricks-genai-inference mlflow langchain langchain-community
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import mlflow 
from databricks import feature_engineering
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.vector_search.client import VectorSearchClient
import json
import requests
import time
from langchain_core.prompts import PromptTemplate
import mlflow.deployments
import pandas as pd
from pyspark.sql.functions import udf, struct, col
from typing import Iterator, Tuple

fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# variables
catalog_name = "speedy_motors"
schema_name = "salesforce"

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

product_feature_table = f"{catalog_name}.{schema_name}.updated_product_table"

# COMMAND ----------

# MAGIC %md ## Create and publish feature table

# COMMAND ----------

from langchain_core.prompts import PromptTemplate
import mlflow.deployments
import pandas as pd
from pyspark.sql.functions import udf, struct, col
from typing import Iterator, Tuple

client = mlflow.deployments.get_deploy_client("databricks")

template = '''Given a car's specifications, write a short description of the car's benefits, advantages, and the personas that would particularly love this type of car.
        
Specifications: Price: {MSRP}, Car Type: {Car_Type}, Color: {Color}, Number of Seats: {Seats}, Fuel Type: {Fuel_Type}, Miles-per-gallon: {MPG}, Drive Train Type: {Drive_Train_Type}, Towing Capacity: {Towing_Capacity}
'''

prompt_template = PromptTemplate(input_variables=["MSRP", "Car_Type", "Color", "Seats", "Fuel_Type", "MPG", "Drive_Train_Type", "Towing_Capacity"],
                                 template=template)

@udf("string")
def get_car_descriptions(*car_features):

    MSRP = car_features[0]
    Car_Type = car_features[1]
    Color = car_features[2]
    Seats = car_features[3]
    Fuel_Type = car_features[4]
    MPG = car_features[5]
    Drive_Train_Type = car_features[6]
    Towing_Capacity = car_features[7]

    prompt = prompt_template.format(MSRP=MSRP, 
                                    Car_Type=Car_Type, 
                                    Color=Color, 
                                    Seats=Seats, 
                                    Fuel_Type=Fuel_Type, 
                                    MPG=MPG,
                                    Drive_Train_Type=Drive_Train_Type,
                                    Towing_Capacity=Towing_Capacity)

    messages=[{
                "role": "system",
                "content": "You are an AI assistant"
            },
            {
                    "role": "user",
                    "content": prompt
            }]

    completions_response = client.predict(
        endpoint="databricks-mixtral-8x7b-instruct",
        inputs={
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 256,
            "n": 1
        }
    )

    # return a string of the specs to be included with the LLM-generated description for unstructured search
    specs_desc = f"Price: {MSRP}, Car Type: {Car_Type}, Color: {Color}, Number of Seats: {Seats}, Fuel Type: {Fuel_Type}, Miles-per-gallon: {MPG}, Drive Train Type: {Drive_Train_Type}, Towing Capacity: {Towing_Capacity}"
    result = f"{specs_desc}\n\n{completions_response.choices[0]['message']['content']}"
    return result

# COMMAND ----------

product_df = spark.sql(f"select * from {product_feature_table}")

car_features = ["MSRP__c", "Car_Type__c", "Color__c", "Seats__c", "Fuel_Type__c", "MPG__c", "Drive_Train_Type__c", "Towing_Capacity__c"]
product_descriptions_df = product_df.select("Id", "Name", *car_features, get_car_descriptions(*car_features).alias("automated_description"))\
                                    .withColumn("MSRP__c", col("MSRP__c").cast("double")) # vector index requires double instead of decimal col type

display(product_descriptions_df)

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

product_feature_table = f"{catalog_name}.{schema_name}.product_ft"

customer_feature_table = fe.create_table(
  name=product_feature_table,
  primary_keys='Id',
  df=product_descriptions_df,
  description='Car features'
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- enable change data feed on the feature table
# MAGIC ALTER TABLE speedy_motors.salesforce.product_ft SET TBLPROPERTIES (delta.enableChangeDataFeed=true);

# COMMAND ----------

# MAGIC %md ### Create a vector index for unstructured data searches

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vector_search_endpoint_name = "speedy_motors_rag_endpoint"

try:
    vsc.create_endpoint(vector_search_endpoint_name)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# vsc.list_endpoints() # use to see all endpoints in a workspace
vsc.list_indexes(vector_search_endpoint_name)

# COMMAND ----------

# MAGIC %md ### Setup Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test out index with a customer preference

# COMMAND ----------

import pprint

car_suggestion = index.similarity_search(
                            query_text="I go skiing every weekend so I want a car that can get me and all my friends to the mountain pretty easily. Not looking to spend too much money though. Price is not a big deal.",
                            columns=["Id", "Name", "MSRP__c", "automated_description"],
                            num_results=2,
                            )

pprint.pp(car_suggestion)

# COMMAND ----------

# MAGIC %md #### Create a vector search index with managed embeddings (using BGE-large)

# COMMAND ----------

feature_table_index = f"{catalog_name}.{schema_name}.speedy_motors_product_feature_index"

try:
  vsc.create_delta_sync_index( 
      endpoint_name=vector_search_endpoint_name,
      index_name=feature_table_index,
      source_table_name=product_feature_table,
      pipeline_type="TRIGGERED",
      primary_key="Id",
      embedding_model_endpoint_name="databricks-bge-large-en", # using the BGE Foundation Model Endpoint
      embedding_source_column="automated_description"
  )
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md #### Wait for the vector search index to be ready

# COMMAND ----------

# no need to run all cells above once endpoint and index are set up
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vector_search_endpoint_name = "speedy_motors_rag_endpoint"
feature_table_index = f"{catalog_name}.{schema_name}.speedy_motors_product_feature_index"

index = vsc.get_index(index_name=feature_table_index, endpoint_name=vector_search_endpoint_name)

# COMMAND ----------

vector_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=feature_table_index)
vector_index.wait_until_ready(verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RAG

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from mlflow.models import infer_signature
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
import string
import re
import mlflow
import json
import pandas as pd
import requests
import time
from databricks.vector_search.client import VectorSearchClient

# vector search endpoint and index 
vsc = VectorSearchClient()
vector_search_endpoint_name = "speedy_motors_rag_endpoint"
feature_table_index = f"{catalog_name}.{schema_name}.speedy_motors_product_feature_index"


index = vsc.get_index(index_name=feature_table_index, endpoint_name=vector_search_endpoint_name)

llm = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=128) 

prompt_template = '''Given a buyer's preferences, write a short description of what the best car is for them to purchase:
                 
                     Buyer's Preferences: {buyer_preference}
                     Car Descriptions: {similarity_search_return}
                  '''

prompt = PromptTemplate(input_variables=["similarity_search_return", "buyer_preference"], 
                        template=prompt_template)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for the buyers preference from the Salesforce account object and provide recommendations for purchasing. 

# COMMAND ----------

first_name = "Alex"
last_name = "Roberts"

buyer_info = spark.sql(f"""
SELECT *
FROM speedy_motors.salesforce.account
WHERE FirstName = '{first_name}' AND LastName = '{last_name}'
""")

buyer_preference = buyer_info.collect()[0]["Preferences__c"]
previous_buyer = buyer_info.collect()[0]["Previous_Buyer__c"]

print("Buyer Preferences: \n")
print(buyer_preference + "\n")

print("Product Recommendation: \n")
chain.invoke({"buyer_preference": buyer_preference, 
              "similarity_search_return": index.similarity_search(
                            query_text=buyer_preference,
                            columns=["Id", "Name", "MSRP__c", "automated_description"],
                            num_results=2,
                            )})

                    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manually enter buyer preferences if they do not have an account on Salesforce

# COMMAND ----------

buyer_preference = '''I have a customer that goes skiing every weekend so they want a car that can get them and all their friends to the mountain pretty easily. Not looking to spend too much money though. Can you also help suggest some accessories I can sell to them?'''

chain.invoke({"buyer_preference": buyer_preference, 
              "similarity_search_return": index.similarity_search(
                            query_text=buyer_preference,
                            columns=["Id", "Name", "MSRP__c", "automated_description"],
                            num_results=2,
                            )})

# COMMAND ----------

# from databricks.feature_store import FeatureLookup

# speedy_motors_feature_endpoint_name = f"speedy_motors_car_features_endpoint"

# # Create a lookup to fetch features by key.
# car_features=[
#   FeatureLookup(
#     table_name=product_feature_table,
#     lookup_key="Id",
#   ),
# ]

# # Create feature spec with the lookup for features
# car_spec_name = f"{catalog_name}.{schema_name}.product2"
# try:
#   fe.create_feature_spec(name=car_spec_name, features=car_features)
# except Exception as e:
#   if "already exists" in str(e):
#     pass
#   else:
#     raise e

# # Create endpoint
# try: 
#   status = fe.create_feature_serving_endpoint(
#     name=speedy_motors_feature_endpoint_name, 
#     config = EndpointCoreConfig(
#       served_entities=ServedEntity(
#         feature_spec_name=car_spec_name, 
#         workload_size="Small", 
#         scale_to_zero_enabled=True
#         )
#     )
#   )
#   print(status)
# except Exception as e:
#   if "already exists" in str(e):
#     pass
#   else:
#     raise e

# COMMAND ----------

# Get endpoint status
# status = fe.get_feature_serving_endpoint(name=speedy_motors_feature_endpoint_name)
# print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC # ********************* TO DO *********************

# COMMAND ----------

# MAGIC %md ### Define a tool to retrieve inventories
# MAGIC
# MAGIC The CustomerRetrievalTool queries the Feature Serving endpoint to serve data from the Databricks online table, thus providing context data based on the user query to the LLM.

# COMMAND ----------

# from langchain.tools import BaseTool
# from typing import Union
# from typing import List
# from databricks.vector_search.client import VectorSearchClient

# class CarRetrievalTool(BaseTool):
#     name = "Cars based on User's Car Preference Vector Server"
#     description = "Use this tool when you need to fetch cars based on a user's preferences."

#     def _run(self, car_preference: str):
#         def calculate_embedding(text):
#             embedding_endpoint_name = "databricks-bge-large-en"
#             url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
#             databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

#             headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
                
#             data = {
#                 "input": text
#             }
#             data_json = json.dumps(data, allow_nan=True)
            
#             print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
            
#             response = requests.request(method='POST', headers=headers, url=url, data=data_json)
#             if response.status_code != 200:
#                 raise Exception(f'Request failed with status {response.status_code}, {response.text}')

#             return response.json()['data'][0]['embedding']
            
#         try:
#             vsc = VectorSearchClient()
#             index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=hotels_table_index)
#             print(index)
#             resp = index.similarity_search(columns=["hotel_id"], query_vector=calculate_embedding(hotel_preference), num_results=5, filters={})
#             print(resp)
#             data_array = resp and resp.get('result', {}).get('data_array')
#             print(data_array)
#         except Exception as e:
#             print(f"Exception while running test case {e}")
#             return []

#         result = [hotel[0] for hotel in data_array]
#         print(result)
#         return result
    
#     def _arun(self, user_id: str):
#         raise NotImplementedError("This tool does not support async")

# COMMAND ----------

# Setup Open API Keys. 
# This allows the notebook to communicate with the ChatGPT conversational model.
# Alternately, you could configure your own LLM model and configure LangChain to refer to it.

OPENAI_API_KEY = dbutils.secrets.get("feature-serving", "OPENAI_API_KEY") #replace this with your openAI API key

# COMMAND ----------

from langchain.agents import initialize_agent
# Tool imports
from langchain.agents import Tool

tools = [
  UserPreferenceTool(),
  CarRetrievalTool(),
]

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Initialize LLM (this example uses ChatOpenAI because later it defines a `chat` agent)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# Initialize agent with tools
aibot = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=5,
    early_stopping_method='force',
    memory=conversational_memory
)

# COMMAND ----------

sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be plan a vacation for the input user_id.

When a customer asks for car suggestions, use the car_preference tool from it's toolbelt to retrieve list of cars that fit in the user's preferences from retrival from the tools.

Overall, Assistant is a powerful system that can help users find the perfect car.
"""

# COMMAND ----------

new_prompt = aibot.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

aibot.agent.llm_chain.prompt = new_prompt

# COMMAND ----------

# MAGIC %md
# MAGIC By incorporating context from the Databricks Lakehouse including online tables and a feature serving endpoint, an AI chatbot created with context retrieval tools performs much better than a generic chatbot. 

# COMMAND ----------

aibot_output = aibot('')

# COMMAND ----------

print(aibot_output['output'])
