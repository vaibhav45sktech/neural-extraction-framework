import pandas as pd
from rdflib import Graph, Namespace
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer

ONTOLOGY_URL = "http://dief.tools.dbpedia.org/server/ontology/dbpedia.owl"
BATCH_SIZE   = 256
MODEL_NAME   = "all-MiniLM-L6-v2"
OUTPUT_DIR   = "."

def fetch_ontology(url):
    print(f"[1/4] Fetching DBpedia ontology from {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    g = Graph()
    g.parse(data=resp.text, format="xml")
    print(f"      Loaded {len(g)} triples into graph.")
    return g

def extract_predicates(g):
    print("[2/4] Extracting predicates via SPARQL ...")
    query = """
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?prop ?label WHERE {
        { ?prop a owl:ObjectProperty }
        UNION
        { ?prop a owl:DatatypeProperty }
        OPTIONAL { ?prop rdfs:label ?label FILTER(lang(?label) = "en") }
    }
    """
    rows = []
    for row in g.query(query):
        uri   = str(row.prop)
        label = str(row.label) if row.label else uri.split("/")[-1].split("#")[-1]
        rows.append({"uri": uri, "label": label})
    df = pd.DataFrame(rows).drop_duplicates(subset="uri")
    print(f"      Found {len(df)} unique predicates.")
    return df

def embed_predicates(df, model_name, batch_size):
    print(f"[3/4] Embedding {len(df)} predicates locally with '{model_name}' ...")
    model  = SentenceTransformer(model_name)
    labels = df["label"].tolist()
    all_embeddings = []
    for start in range(0, len(labels), batch_size):
        batch = labels[start : start + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(vecs)
        done = min(start + batch_size, len(labels))
        print(f"      Embedded {done}/{len(labels)}", end="\r")
    print()
    return np.vstack(all_embeddings)

def save_outputs(df, embeddings, out_dir):
    print("[4/4] Saving outputs ...")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    df[["uri"]].to_csv(os.path.join(out_dir, "predicates.csv"), index=False)
    df[["label"]].to_csv(os.path.join(out_dir, "predicate_labels.csv"), index=False)
    print(f"      embeddings.npy  → shape {embeddings.shape}")
    print(f"      predicates.csv  → {len(df)} rows")
    print("\n✅  Done! You can now run NEF.py with these files.")

if __name__ == "__main__":
    g          = fetch_ontology(ONTOLOGY_URL)
    df         = extract_predicates(g)
    embeddings = embed_predicates(df, MODEL_NAME, BATCH_SIZE)
    save_outputs(df, embeddings, OUTPUT_DIR)
