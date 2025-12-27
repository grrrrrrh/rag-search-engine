from cli.lib.keyword_search import InvertedIndex, tokenize_text

idx = InvertedIndex()
idx.load()

query = "space adventure"
tokens = tokenize_text(query)
print("Query tokens:", tokens)

for t in tokens:
    docs_with_t = len(idx.index.get(t, []))
    print(f"Token '{t}' appears in {docs_with_t} documents")

print("Total docs:", len(idx.docmap))