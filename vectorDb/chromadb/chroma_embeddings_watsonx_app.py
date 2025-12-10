import os
import numpy as np
import pandas as pd
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain_ibm import WatsonxLLM

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(app_name="chroma_embeddings_watsonx_app")

embedding_function = SentenceTransformerEmbeddingFunction()

@task(name="watsonx_llm_init")
def watsonx_llm_init() -> WatsonxLLM:
    watsonx_llm_parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 10,
        GenTextParamsMetaNames.TEMPERATURE: 0.7,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }
    model = 'ibm/granite-4-h-small'
    watsonx_llm = WatsonxLLM(
        model_id=model,
        url="https://us-south.ml.cloud.ibm.com",
        apikey="<watsonx-api-key>",
        project_id="<watsonx-project-id>",
        params=watsonx_llm_parameters,
    )
    return watsonx_llm


chroma_client = chromadb.Client()

embeddings_collection = chroma_client.create_collection(
    name="embeddings_demo",
    embedding_function=embedding_function
)


@task(name="generate_embeddings")
def generate_embeddings(texts):
    try:
        embeddings = embedding_function(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        return embeddings
    except Exception as e:
        print(f"✗ Error generating embeddings: {e}")
        return None


@task(name="compute_cosine_similarity")
def compute_cosine_similarity(embedding1, embedding2):
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    except Exception as e:
        print(f"✗ Error computing similarity: {e}")
        return None


@task(name="compute_euclidean_distance")
def compute_euclidean_distance(embedding1, embedding2):
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        distance = np.linalg.norm(vec1 - vec2)
        return float(distance)
    except Exception as e:
        print(f"✗ Error computing distance: {e}")
        return None


@workflow("add_documents_with_embeddings")
def add_documents_with_embeddings(ids, documents, metadatas=None, embeddings=None):
    try:
        embeddings_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"✓ Added {len(ids)} documents with embeddings")
        return {"status": "success", "count": len(ids)}
    except Exception as e:
        print(f"✗ Error adding documents: {e}")
        return {"status": "error", "message": str(e)}


@workflow("query_by_embedding")
def query_by_embedding(query_embeddings, n_results=5, where=None):
    try:
        result = embeddings_collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        print(f"✓ Query returned {len(result['ids'][0])} results")
        return result
    except Exception as e:
        print(f"✗ Error querying by embedding: {e}")
        return {"status": "error", "message": str(e)}


@workflow("get_embeddings")
def get_embeddings(ids=None, where=None, limit=None):
    try:
        result = embeddings_collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=["documents", "metadatas", "embeddings"]
        )
        print(f"✓ Retrieved {len(result['ids'])} documents with embeddings")
        return result
    except Exception as e:
        print(f"✗ Error getting embeddings: {e}")
        return {"status": "error", "message": str(e)}


@workflow("semantic_search")
def semantic_search(query_text, n_results=5):
    try:
        result = embeddings_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\n🔍 Semantic Search Results for: '{query_text}'")
        print("=" * 80)
        
        for i, (doc_id, doc, distance) in enumerate(zip(
            result['ids'][0],
            result['documents'][0],
            result['distances'][0]
        ), 1):
            similarity = 1 - distance
            print(f"\n{i}. ID: {doc_id}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   Document: {doc[:100]}...")
        
        return result
    except Exception as e:
        print(f"✗ Error in semantic search: {e}")
        return {"status": "error", "message": str(e)}


@workflow("compare_embeddings")
def compare_embeddings(text1, text2):
    try:
        embeddings = generate_embeddings([text1, text2])
        
        if embeddings is None:
            return None
        
        cosine_sim = compute_cosine_similarity(embeddings[0], embeddings[1])
        euclidean_dist = compute_euclidean_distance(embeddings[0], embeddings[1])
        
        print(f"\n📊 Embedding Comparison")
        print("=" * 80)
        print(f"Text 1: {text1[:60]}...")
        print(f"Text 2: {text2[:60]}...")
        print(f"\nCosine Similarity: {cosine_sim:.4f}")
        print(f"Euclidean Distance: {euclidean_dist:.4f}")
        
        return {
            "cosine_similarity": cosine_sim,
            "euclidean_distance": euclidean_dist,
            "embeddings": embeddings
        }
    except Exception as e:
        print(f"✗ Error comparing embeddings: {e}")
        return None


@workflow("find_similar_documents")
def find_similar_documents(document_id, n_results=5):
    try:
        ref_doc = embeddings_collection.get(
            ids=[document_id],
            include=["embeddings", "documents"]
        )
        
        if not ref_doc['ids']:
            print(f"✗ Document {document_id} not found")
            return None
        
        ref_embedding = ref_doc['embeddings'][0]
        ref_text = ref_doc['documents'][0]
        
        result = query_by_embedding(
            query_embeddings=[ref_embedding],
            n_results=n_results + 1
        )
        
        print(f"\n🔗 Documents Similar to '{document_id}'")
        print(f"Reference: {ref_text[:60]}...")
        print("=" * 80)
        
        for i, (doc_id, doc, distance) in enumerate(zip(
            result['ids'][0][1:],
            result['documents'][0][1:],
            result['distances'][0][1:]
        ), 1):
            similarity = 1 - distance
            print(f"\n{i}. ID: {doc_id}")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   Document: {doc[:80]}...")
        
        return result
    except Exception as e:
        print(f"✗ Error finding similar documents: {e}")
        return None


@workflow("analyze_embedding_clusters")
def analyze_embedding_clusters(category_field="category"):
    try:
        all_docs = embeddings_collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not all_docs['ids']:
            print("✗ No documents in collection")
            return None
        
        categories = {}
        for doc_id, doc, metadata, embedding in zip(
            all_docs['ids'],
            all_docs['documents'],
            all_docs['metadatas'],
            all_docs['embeddings']
        ):
            category = metadata.get(category_field, "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'id': doc_id,
                'document': doc,
                'embedding': embedding
            })
        
        print(f"\n📈 Embedding Cluster Analysis")
        print("=" * 80)
        
        watsonx_llm = watsonx_llm_init()
        
        for category, docs in categories.items():
            print(f"\n📁 Category: {category}")
            print(f"   Documents: {len(docs)}")
            
            if len(docs) > 1:
                similarities = []
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        sim = compute_cosine_similarity(
                            docs[i]['embedding'],
                            docs[j]['embedding']
                        )
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                print(f"   Avg Intra-cluster Similarity: {avg_similarity:.4f}")
                
                sample_docs = [d['document'][:100] for d in docs[:3]]
                prompt = f"""Analyze these documents from the '{category}' category and provide a brief summary of their common theme:

{chr(10).join(f'{i+1}. {doc}' for i, doc in enumerate(sample_docs))}

Summary:"""
                
                summary = watsonx_llm.invoke(prompt)
                print(f"   Theme: {summary.strip()}")
        
        return categories
    except Exception as e:
        print(f"✗ Error analyzing clusters: {e}")
        return None


@workflow(name="demo_embedding_operations")
def demo_embedding_operations():
    print("\n" + "="*80)
    print("CHROMADB EMBEDDINGS DEMO WITH WATSONX")
    print("="*80)
    
    sample_docs = [
        ("ai_1", "Machine learning algorithms learn patterns from data.", {"category": "AI/ML", "topic": "learning"}),
        ("ai_2", "Neural networks are inspired by biological neurons.", {"category": "AI/ML", "topic": "architecture"}),
        ("ai_3", "Deep learning models require large amounts of training data.", {"category": "AI/ML", "topic": "training"}),
        
        ("health_1", "Medical imaging helps diagnose diseases early.", {"category": "Healthcare", "topic": "diagnosis"}),
        ("health_2", "Electronic health records improve patient care coordination.", {"category": "Healthcare", "topic": "records"}),
        ("health_3", "Telemedicine enables remote patient consultations.", {"category": "Healthcare", "topic": "telehealth"}),
        
        ("tech_1", "Cloud computing provides scalable infrastructure.", {"category": "Technology", "topic": "cloud"}),
        ("tech_2", "Blockchain technology ensures data immutability.", {"category": "Technology", "topic": "blockchain"}),
        ("tech_3", "Quantum computers use quantum mechanics principles.", {"category": "Technology", "topic": "quantum"}),
    ]
    
    print("\n1. ADDING DOCUMENTS WITH AUTO-GENERATED EMBEDDINGS")
    print("-" * 80)
    ids = [doc[0] for doc in sample_docs]
    documents = [doc[1] for doc in sample_docs]
    metadatas = [doc[2] for doc in sample_docs]
    
    add_documents_with_embeddings(ids, documents, metadatas)
    
    print("\n2. COMPARING EMBEDDINGS")
    print("-" * 80)
    compare_embeddings(
        "Machine learning algorithms learn from data",
        "Neural networks process information like the brain"
    )
    
    compare_embeddings(
        "Machine learning algorithms learn from data",
        "Cloud computing provides scalable resources"
    )
    
    print("\n3. SEMANTIC SEARCH")
    print("-" * 80)
    semantic_search("artificial intelligence and neural networks", n_results=3)
    
    print("\n4. FINDING SIMILAR DOCUMENTS")
    print("-" * 80)
    find_similar_documents("ai_1", n_results=3)
    
    print("\n5. QUERY BY CUSTOM EMBEDDING")
    print("-" * 80)
    custom_query = "How does AI help in medical diagnosis?"
    custom_embedding = generate_embeddings([custom_query])
    if custom_embedding:
        result = query_by_embedding(custom_embedding, n_results=3)
        print(f"\nTop results for: '{custom_query}'")
        for i, (doc_id, doc) in enumerate(zip(result['ids'][0], result['documents'][0]), 1):
            print(f"{i}. {doc_id}: {doc[:60]}...")
    
    print("\n6. ANALYZING EMBEDDING CLUSTERS")
    print("-" * 80)
    analyze_embedding_clusters(category_field="category")
    
    print("\n7. RETRIEVING EMBEDDINGS")
    print("-" * 80)
    result = get_embeddings(ids=["ai_1", "health_1"], limit=2)
    if "embeddings" in result:
        for doc_id, embedding in zip(result['ids'], result['embeddings']):
            print(f"   {doc_id}: Embedding dimension = {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
    
    print("\n" + "="*80)
    print("EMBEDDINGS DEMO COMPLETED")
    print("="*80 + "\n")


@workflow(name="demo_advanced_embedding_analysis")
def demo_advanced_embedding_analysis():
    print("\n" + "="*80)
    print("ADVANCED EMBEDDING ANALYSIS WITH WATSONX")
    print("="*80)
    
    all_docs = embeddings_collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    if not all_docs['ids']:
        print("✗ No documents in collection. Run demo_embedding_operations first.")
        return
    
    print("\n1. IDENTIFYING OUTLIER DOCUMENTS")
    print("-" * 80)
    
    embeddings = np.array(all_docs['embeddings'])
    centroid = np.mean(embeddings, axis=0)
    
    distances = [compute_euclidean_distance(emb, centroid) for emb in embeddings]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    outliers = []
    for i, (doc_id, doc, dist) in enumerate(zip(all_docs['ids'], all_docs['documents'], distances)):
        if dist > mean_dist + 1.5 * std_dist:
            outliers.append((doc_id, doc, dist))
            print(f"   Outlier: {doc_id}")
            print(f"   Distance from centroid: {dist:.4f}")
            print(f"   Document: {doc[:60]}...")
    
    if outliers:
        print("\n2. WATSONX ANALYSIS OF OUTLIERS")
        print("-" * 80)
        watsonx_llm = watsonx_llm_init()
        
        for doc_id, doc, dist in outliers[:2]:
            prompt = f"""Explain why this document might be semantically different from others in a collection about AI, Healthcare, and Technology:

Document: {doc}

Explanation:"""
            
            explanation = watsonx_llm.invoke(prompt)
            print(f"\n   {doc_id}: {explanation.strip()}")
    
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo_embedding_operations()
    
    demo_advanced_embedding_analysis()