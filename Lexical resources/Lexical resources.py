import nltk
from nltk.corpus import wordnet as wn
from itertools import product
from scipy.stats import spearmanr

# nltk.download('wordnet')

with open("sample4.csv", encoding="utf-8") as file:
    triples = [line.strip().split(',') for line in file.readlines()]
    score_map = {tuple(triple[:2]): float(triple[2]) for triple in triples}
print(triples)
for w1, w2 in list(score_map)[:2]:
    print("\nWords: %s-%s\nGround truth score: %.2f" % (w1, w2, score_map[(w1, w2)]))

    ss1 = wn.synset(w1 + ".n.01")
    ss2 = wn.synset(w2 + ".n.01")

    print("\nPath: %.3f" % ss1.path_similarity(ss2), end=' ')
    print("\nwup: %.3f" % ss1.wup_similarity(ss2), end=' ')
    print("\nshortest_path: %.3f" % ss1.shortest_path_distance(ss2))

list_pairs = list(score_map)
wup_list, true_list, path_list, lch_list = [], [], [], []

for w1, w2 in list_pairs:

    all_w1 = wn.synsets(w1, pos='n')
    all_w2 = wn.synsets(w2, pos="n")

    wup = max([it1.wup_similarity(it2) for it1, it2 in product(all_w1, all_w2)])
    wup_list.append(wup)

    path = max([it1.path_similarity(it2) for it1, it2 in product(all_w1, all_w2)])
    path_list.append(path)

    lch = max([it1.lch_similarity(it2) for it1, it2 in product(all_w1, all_w2)])
    lch_list.append(lch)

    true_list.append(score_map[(w1, w2)])

coef, p = spearmanr(wup_list, true_list)
print("wup Spearman R: %.4f" % coef)

coef, p1 = spearmanr(path_list, true_list)
print("path Spearman R: %.4f" % coef)

coef1, p2 = spearmanr(lch_list, true_list)
print("lch Spearman R: %.4f" % coef1)

s1 = wn.synset("wood" + ".n.01")
print(f'Количество гипонимов к слову wood.n.01: {len(s1.hyponyms())},\nЗначение первого гипонима: {s1.hyponyms()[0]}')