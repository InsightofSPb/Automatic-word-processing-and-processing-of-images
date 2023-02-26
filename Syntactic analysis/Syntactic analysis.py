import sys
from conllu import parse  # парсер, создающий вложенный словарь
from graphviz import Digraph, Source  # для отрисовки деревьев
from ufal.udpipe import Model, Pipeline, ProcessingError  # UDPipe
import wget



# Скачиваем необходимые для дальнейшей работы материалы

# url1 = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-syntagrus-ud-2.5-191206.udpipe'
# url2 = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-taiga-ud-2.5-191206.udpipe'
#
# file1 = wget.download(url1)
# file1 = wget.download(url2)


# Функция для загрузки моделей на основе текстов. Обучение
def loading():
    syntagrus_model_file = "russian-syntagrus-ud-2.5-191206.udpipe"  # корпус русского языка с человеческими правками
    taiga_model_file = 'russian-taiga-ud-2.5-191206.udpipe'  # набор текстов от Вышки, учитывающих то, чего нет в Syntagrus

    print("Loading model:" + syntagrus_model_file)
    syntagrus_model = Model.load(syntagrus_model_file)
    print("Loading done")

    print("Loading model:" + taiga_model_file)
    taiga_model = Model.load(taiga_model_file)
    print("Loading done")
    return syntagrus_model, taiga_model


syn, taig = loading()
input_format = "tokenize"
output_format = "conllu"

# Pipeline(given model, input, tagger, parser, output)
# In general, pipeline - a sequence of data processing mechanisms
# Применяем обработку к сырому тексту
syntag_pipeline = Pipeline(syn, input_format, Pipeline.DEFAULT, Pipeline.DEFAULT, output_format)
taiga_pipeline = Pipeline(taig, input_format, Pipeline.DEFAULT, Pipeline.DEFAULT, output_format)

error = ProcessingError()


# Применяем цепочки обработки к сырому тексту. Первая переменная - предложение, второя - цепочка обработки
def parse_syntax(raw_text, pipeline):
    """ Applying UDPipe """
    text = raw_text.strip()  # убираем лишние пробелы
    processed = pipeline.process(text, error)
    # Process the given input stream, writing results to the given output stream. If the processing succeeded, true
    # is returned; otherwise, false is returned with an error stored in the error argument.
    if error.occurred():
        sys.stderr.write("An error occurred when running run_udpipe: ")  # сообщение об ошибках
        sys.stderr.write(error.message)
        sys.stderr.write("\n")
    return processed


# Проводим разбор предложений
syntag = parse_syntax("В отличие от Human Genome Project, PGP не предполагает расшифровку ДНК участников целиком.", syntag_pipeline)
taiga = parse_syntax("В отличие от Human Genome Project, PGP не предполагает расшифровку ДНК участников целиком.", taiga_pipeline)
print(syntag)


def parse_output(conll):
    sentences = parse(conll)
    return [s.to_tree() for s in sentences]  # to.tree() - turning TokenList to TokenTree


class Root(object):
    def __init__(self, child):
        self.token = {"form": "", "id": 0, "deprel": "root"}
        self.children = [child]


def draw_tree(tree):
    g = Digraph(format='png')  # node - лепесток, edge -

    queue = [Root(tree)]  # очередь по типу FIFO, для обхода по дереву

    while len(queue) > 0:
        head = queue.pop()
        head_label = head.token["form"] + " (%d)" % head.token["id"]
        queue.extend(head.children)


        for c in head.children:
            c_label = c.token["form"] + " (%d)" % c.token["id"]
            g.edge(head_label, c_label, label=c.token["deprel"])

    return g


str_tree = parse_output(syntag.strip())[0]
taiga_tree = parse_output(taiga.strip())[0]


def edges_sets(tree):
    """ Reading edges encoded as strings """

    edges_labeled = []
    edges_unlabeled = []
    queue = [Root(tree)]

    while len(queue) > 0:
        head = queue.pop()
        head_label = head.token["form"] + " (%d)" % head.token["id"]
        queue.extend(head.children)

        for c in head.children:
            c_label = c.token["form"] + " (%d)" % c.token["id"]
            edges_unlabeled.append((head_label, c_label))
            edges_labeled.append((head_label, c_label, c.token["deprel"]))

    return set(edges_labeled), set(edges_unlabeled)


def accuracy(gold, pred):
  gold_labeled, gold_unlabeled = edges_sets(gold)
  pred_labeled, pred_unlabeled = edges_sets(pred)

  return len(gold_labeled.intersection(pred_labeled)) / len(gold_labeled),\
         len(gold_unlabeled.intersection(pred_unlabeled)) / len(gold_unlabeled)


LAS, UAS = accuracy(str_tree, taiga_tree)
print("LAS: %.4f UAS: %.4f" % (LAS, UAS))

t = []
for i in range(2):
    depth = 0
    if i == 0:
        queue = [(taiga_tree, depth)]
        while len(queue) > 0:
            head, depth = queue.pop(0)
            queue.extend([(c, depth + 1) for c in head.children])
    if i == 1:
        queue = [(str_tree, depth)]
        while len(queue) > 0:
            head, depth = queue.pop(0)
            queue.extend([(c, depth + 1) for c in head.children])
    t.append(depth)
print("Глубина дерева: ", max(t))