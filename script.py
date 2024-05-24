import pysrt
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import words

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

nltk.download('words')
common_words = set(words.words())
def read_subtitle(file_path):
    return pysrt.open(file_path)

def normalize_name(name):
    return re.sub(r'\W+', '', name).lower()

def extract_names_and_speakers(subs):
    dialogues = []
    for sub in subs:
        text = sub.text
        if text.startswith("♪"):
            dialogues.append(["music"])
        else:
            doc = nlp(text)
            dialogue_names = []
            for ent in doc.ents:
                if (ent.label_ == "PERSON" or ent.label_ == "PROPN") and normalize_name(ent.text) not in common_words:
                    normalized_name = normalize_name(ent.text)
                    dialogue_names.append(normalized_name)
            if dialogue_names:
                dialogues.append(dialogue_names)
    return dialogues


def build_character_graph(dialogues):
    G = nx.DiGraph()
    for dialogue in dialogues:
        for i, name in enumerate(dialogue):
            if not G.has_node(name):
                G.add_node(name)
            for j in range(i + 1, len(dialogue)):
                G.add_edge(name, dialogue[j], weight=G.get_edge_data(name, dialogue[j], {'weight': 0})['weight'] + 1)
                G.add_edge(dialogue[j], name, weight=G.get_edge_data(dialogue[j], name, {'weight': 0})['weight'] + 1)
    return G

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}%' if pct >= 2 else ''
    return my_format

def plot_pie_chart(rankings):
    # Sort rankings by value (PageRank score)
    sorted_rankings = dict(sorted(rankings.items(), key=lambda item: item[1]))

    labels = list(sorted_rankings.keys())
    sizes = list(sorted_rankings.values())

    fig, ax = plt.subplots(figsize=(12, 7))
    wedges, texts, autotexts = ax.pie(sizes, autopct=autopct_format(sizes), startangle=140, pctdistance=0.85)

    # Add numbering to the chart
    for i, a in enumerate(autotexts):
        a.set_text(f'{i+1}\n' + a.get_text())

    # Determine the number of columns for the legend
    num_columns = (len(labels) + 9) // 30  # Adjust the divisor to control the number of columns

    plt.legend(wedges, [f'{i+1}. {label}' for i, label in enumerate(labels)], title="Characters", loc="center left", bbox_to_anchor=(1, 0.5), ncol=num_columns)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title("Character Importance Based on PageRank")
    plt.subplots_adjust(left=0.1, right=0.7)  # Adjust the position of the pie chart and legend
    plt.show()
def main(subtitle_file):
    subs = read_subtitle(subtitle_file)
    dialogues = extract_names_and_speakers(subs)
    G = build_character_graph(dialogues)
    pr = nx.pagerank(G, weight='weight')
    plot_pie_chart(pr)

if __name__ == "__main__":
    subtitle_file = "Enter file path (with / ):"  # Replace with your subtitle file path
    main(subtitle_file)
