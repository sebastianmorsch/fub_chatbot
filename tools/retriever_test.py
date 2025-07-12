from retriever import Retriever

# Initialize and load or rebuild the index
retriever = Retriever()
retriever.load_or_build(force_rebuild=True)

# Example question (adjust here!)
query = "Wie schalte ich Studio 7 ein?"

print(f"\nQuery: {query}\n")

results = retriever.search(query, k=5)

print("Relevant chunks:\n")
for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---\n{result}\n")
