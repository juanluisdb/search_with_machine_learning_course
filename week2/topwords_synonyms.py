import fasttext
import csv

model = fasttext.load_model("/workspace/datasets/fasttext/title_model_v2.bin")
synonyms = []
threshold = 0.75

with open("/workspace/datasets/fasttext/top_words.txt", "r") as top_words:
    for word in top_words:
        neighbors = [
            synonym
            for sim, synonym in model.get_nearest_neighbors(word)
            if sim >= threshold
        ]
        if neighbors:
            synonyms.append(neighbors)

with open("/workspace/datasets/fasttext/synonyms.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(synonyms)