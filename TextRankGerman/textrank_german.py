import os
import sys
import copy
import collections

import nltk
import nltk.tokenize
from HanTa import HanoverTagger as ht

sys.path.append(".")

import pagerank

def __is_punctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-"]

def __preprocess_document(document, relevant_pos_tags):
    '''
    This function accepts a string representation 
    of a document as input, and returns a tokenized
    list of words corresponding to that document.
    '''

    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    words = nltk.word_tokenize(document, language="german")

    # lemmatisation and POS tagging
    # lemmata = tagger.tag_sent(words, taglevel=1)
    # words = [x[1] for x in lemmata]
    # pos_tags = [x[2] for x in lemmata]

    # just POS tagging
    pos_tags = tagger.tag_sent(words, taglevel=0)
    
    # Filter out words with irrelevant POS tags
    filtered_words = []
    for index, word in enumerate(words):
        word = word.lower()
        tag = pos_tags[index]
        if not __is_punctuation(word) and tag in relevant_pos_tags:
            filtered_words.append(word)

    return filtered_words


def textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"]):
    '''
    This function accepts a string representation
    of a document and three hyperperameters as input.
    It returns Pandas matrix (that can be treated
    as a dictionary) that maps words in the
    document to their associated TextRank significance
    scores. Note that only words that are classified
    as having relevant POS tags are present in the
    map.
    '''
    
    # Tokenize document:
    words = __preprocess_document(document, relevant_pos_tags)
    
    # Build a weighted graph where nodes are words and
    # edge weights are the number of times words cooccur
    # within a window of predetermined size. In doing so
    # we double count each coocurrence, but that will not
    # alter relative weights which ultimately determine
    # TextRank scores.
    edge_weights = collections.defaultdict(lambda: collections.Counter())
    for index, word in enumerate(words):
        for other_index in range(index - window_size, index + window_size + 1):
            if other_index >= 0 and other_index < len(words) and other_index != index:
                other_word = words[other_index]
                edge_weights[word][other_word] += 1.0

    # Apply PageRank to the weighted graph:
    word_probabilities = pagerank.power_iteration(edge_weights, rsp=rsp)
    word_probabilities.sort_values(ascending=False)

    return word_probabilities


def apply_text_tank(file_name, title="a document"):
    print()
    print("Reading \"%s\" ..." % title)
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    document = open(file_path, encoding="utf8").read()
    #document = __ascii_only(document)
    
    print("Applying TextRank to \"%s\" ..." % title)
    keyword_scores = textrank(document)
    
    print()
    header = "Keyword Significance Scores for \"%s\":" % title
    print(header)
    print("-" * len(header))
    print(keyword_scores)
    print()


def main():
    apply_text_tank("story_from_chatgpt.txt", title="random story")


if __name__ == "__main__":
    main()