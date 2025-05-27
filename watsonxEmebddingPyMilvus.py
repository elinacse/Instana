from pymilvus import MilvusClient

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

from langchain_ibm.embeddings import WatsonxEmbeddings 
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
import os

Traceloop.init(app_name="Watsonx_Embeddings_MilvusClient")

# connect to Milvus Locally
@task(name="setup_milvus_client")
def setup_milvus_client(uri: str, collection_name: str, dimension: int):
    client = MilvusClient(uri=uri)
    ##create a collection in Milvus DB
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name, dimension=dimension, timeout=10, metric_type="COSINE"
    )
    return client


embedding_model = None  # Define Embedding Model globally

#Initialize watsonx embedding model 
@task(name="initialize_embedding_model")
def initialize_embedding_model(
    ibm_cloud_url: str,
    ibm_cloud_api_key: str,
    model_id: str,
    project_id: str,
    model_kwargs: dict = None,
    encode_kwargs: dict = None,
):
    embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    global embedding_model
    model_kwargs = model_kwargs or {}
    encode_kwargs = encode_kwargs or {"normalize_embeddings": False}

    embedding_model = WatsonxEmbeddings(
        url=ibm_cloud_url,
        project_id=project_id,
        model_id=model_id,
        apikey=ibm_cloud_api_key,
        params=embed_params
    )


# embed Documents and insert it to Milvus DB
@task(name="encode_documents_and_insert")
def encode_documents_and_insert(
    client: MilvusClient,
    collection_name: str,
    partition_name: str,
    docs: list,
    subject: str,
    timeout: float,
):
    vectors = embedding_model.embed_documents(docs)
    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": subject}
        for i in range(len(vectors))
    ]

    res = client.insert(
        collection_name=collection_name,
        partition_name=partition_name,
        data=data,
        timeout=timeout,
    )
    print(res)


# apply vector embedding on the query and search the same in the vecotr db
@task(name="perform_vector_search")
def perform_vector_search(
    client: MilvusClient,
    collection_name: str,
    query: str,
    limit: int,
    output_fields: list,
):
    query_vector = embedding_model.embed_query(query)
    result = client.search(
        collection_name=collection_name,
        partition_name="partitionA",
        data=[query_vector],
        limit=limit,
        output_fields=output_fields,
    )
    return result


# search in vecotr db with filters applied
@task(name="perform_vector_search_with_filter")
def perform_vector_search_with_filter(
    client: MilvusClient,
    collection_name: str,
    partition_names: list,
    anns_field: str,
    search_params: dict,
    query: str,
    filter: str,
    limit: int,
    output_fields: list,
    timeout: float,
):
    query_vector = embedding_model.embed_query(query)
    searchResult = client.search(
        collection_name=collection_name,
        partition_names=partition_names,
        search_params=search_params,
        anns_field=anns_field,
        data=[query_vector],
        filter=filter,
        limit=limit,
        output_fields=output_fields,
        timeout=timeout,
    )
    return searchResult


# query for entries in the Collection 
@task(name="perform_query")
def perform_query(
    client: MilvusClient,
    collection_name: str,
    filter: str,
    output_fields: list,
):
    queryResult = client.query(
        collection_name=collection_name,
        filter=filter,
        partition_names=["partitionA"],
        output_fields=output_fields,
    )
    return queryResult


# query the db passing list of ids
@task(name="perform_query_ids")
def perform_query_Ids_partition(
    client: MilvusClient,
    collection_name: str,
    partition_names: list,
    limit: int,
    ids: list,
    output_fields: list,
    timeout: float,
):
    queryResult = client.query(
        collection_name=collection_name,
        partition_names=partition_names,
        limit=limit,
        ids=ids,
        timeout=timeout,
    )
    return queryResult

# delete entries from the collection
@task(name="delete_entities")
def delete_entities(
    client: MilvusClient,
    collection_name: str,
    partition_name: str,
    ids: list = None,
    filter: str = None,
    timeout: float = None,
):
    if ids is not None:
        deleteResult = client.delete(collection_name=collection_name, ids=ids)
        print(deleteResult)
    if filter is not None:
        deleteRes = client.delete(
            collection_name=collection_name,
            timeout=timeout,
            filter=filter,
            partition_name=partition_name,
        )
        print(deleteRes)

# modify data in the collection
@task(name="upsert_entities")
def upsert_entities(
    client: MilvusClient,
    collection_name: str,
    partition_name: str,
    docs: list,
    ids: list,
    subject: str,
    timeout: float,
):
    vectors = embedding_model.embed_documents(docs)
    data = [
        {"id": ids[i], "vector": vectors[i], "text": docs[i], "subject": subject}
        for i in range(len(vectors))
    ]

    res = client.upsert( 
        collection_name=collection_name,
        partition_name=partition_name,
        data=data,
        timeout=timeout,
    )
    print("Upsert Result:", res)


@task(name="get_entities")
def get_entities(
    client: MilvusClient,
    collection_name: str,
    partition_names: list,
    output_fields: list,
    ids: list,
    timeout: float,
):
    result = client.get(
        collection_name=collection_name,
        partition_names=partition_names,
        output_fields=output_fields,
        ids=ids,
        timeout=timeout,
    )
    return result

@workflow(name="milvus_operations_with_watsonx")  
def milvus_operations_with_watsonx():
    client = setup_milvus_client(
        uri="http://127.0.0.1:19530", collection_name="demo_collection", dimension=768
    )
    partition_name = "partitionA"
    client.create_partition(
        collection_name="demo_collection", partition_name=partition_name
    )

    #  Watsonx Embedding model parameters
    ibm_cloud_url = os.getenv("WATSONX_URL")
    ibm_cloud_api_key = os.getenv("WATSONX_API_KEY")
    model_id = (
        "ibm/slate-125m-english-rtrvr"  # or any other supported model
    )
    project_id=os.getenv("WATSONX_PROJECT_ID")

    initialize_embedding_model(
        ibm_cloud_url=ibm_cloud_url,
        ibm_cloud_api_key=ibm_cloud_api_key,
        model_id=model_id,
        project_id=project_id
    )

    docs_history = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    encode_documents_and_insert(
        client=client,
        collection_name="demo_collection",
        partition_name=partition_name,
        docs=docs_history,
        subject="history",
        timeout=10,
    )  

    # Upsert example
    new_docs_history = [
        "Alan Turing developed the Turing Test.",
        "Artificial intelligence continues to evolve.",
    ]
    new_ids_history = [
        0,
        1,
    ]  
    upsert_entities(
        client=client,
        collection_name="demo_collection",
        partition_name=partition_name,
        docs=new_docs_history,
        ids=new_ids_history,
        subject="history",
        timeout=10,
    )

    # Get example
    get_result = get_entities(
        client=client,
        collection_name="demo_collection",
        partition_names=[partition_name],
        output_fields=["text", "subject"],
        ids=new_ids_history,
        timeout=10,
    )
    print("Get Result:", get_result)

    # Semantic Search
    # Vector search
    result = perform_vector_search(
        client=client,
        collection_name="demo_collection",
        query="Who is Alan Turing?",
        limit=2,
        output_fields=["text", "subject"],
    )
    print(result)

    # Vector Search with Metadata Filtering
    docs_biology = [
        "Machine learning has been used for drug design.",
        "Computational synthesis with AI algorithms predicts molecular properties.",
        "DDR1 is involved in cancers and fibrosis.",
    ]

    encode_documents_and_insert(
        client=client,
        collection_name="demo_collection",
        partition_name=partition_name,
        docs=docs_biology,
        subject="biology",
        timeout=10,
    )

    search_params = {"metric_type": "COSINE", "params": {}}

    searchResult = perform_vector_search_with_filter(
        client=client,
        collection_name="demo_collection",
        partition_names=[partition_name],
        anns_field="vector",
        search_params=search_params,
        query="tell me AI related information",
        filter="subject == 'biology'",
        limit=2,
        output_fields=["text", "subject"],
        timeout=10,
    )
    print(searchResult)

    # Perform Query
    queryResult = perform_query(
        client=client,
        collection_name="demo_collection",
        filter="subject == 'history'",
        output_fields=["text", "subject"],
    )
    print(queryResult)

    # Perform Query with ids as input param
    queryResult = perform_query_Ids_partition(
        client=client,
        collection_name="demo_collection",
        partition_names=[partition_name],
        limit=1,
        ids=[0, 2],
        output_fields=["text", "subject"],
        timeout=10,
    )
    print(queryResult)

    # Delete entities
    delete_entities(
        client=client,
        collection_name="demo_collection",
        partition_name=partition_name,
        ids=[0, 2],
        timeout=10,
    )

    # 8. Delete entities by a filter expression
    delete_entities(
        client=client,
        collection_name="demo_collection",
        partition_name=partition_name,
        filter="subject == 'biology'",
        timeout=10,
    )


milvus_operations_with_watsonx()
