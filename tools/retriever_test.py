from retriever import Retriever

# Initialisieren und Index laden oder neu bauen
retriever = Retriever()
retriever.load_or_build(force_rebuild=True)

# Beispiel-Frage (hier anpassen!)
query = "Wie schalte ich Studio 7 ein?"

print(f"\nQuery: {query}\n")

results = retriever.search(query, k=5)

print("Relevant chunks:\n")
for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---\n{result}\n")
