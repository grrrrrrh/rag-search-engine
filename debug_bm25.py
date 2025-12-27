from inverted_index import InvertedIndex
from search_utils import tokenize

idx = InvertedIndex.load()
avgdl = sum(idx.doc_lengths.values()) / len(idx.doc_lengths)
print("N docs:", len(idx.docmap))
print("avgdl:", avgdl)

def explain(query: str):
    q = tokenize(query)
    print("\nQUERY:", query)
    print("TOKENS:", q)
    results = idx.bm25_search(query, limit=5)
    for rank, (doc_id, total) in enumerate(results, 1):
        title = idx.docmap[doc_id].get("title", "")
        dl = idx.doc_lengths.get(doc_id, 0)
        print(f"\n{rank}. ({doc_id}) {title}")
        print("  dl:", dl, " total:", total)
        for tok in q:
            tf = idx._tf_token(doc_id, tok)
            df = len(idx.index.get(tok, set()))
            idf = idx._bm25_idf_token(tok)
            tfpart = idx._bm25_tf_token(doc_id, tok)
            termscore = tfpart * idf
            print(f"  tok='{tok}' tf={tf} df={df} idf={idf:.6f} bm25_tf={tfpart:.6f} term={termscore:.6f}")

explain("space adventure")
explain("animated family")
