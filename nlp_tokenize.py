import os
import numpy as np
import dill as pickle
import itertools
import copy

from nltk import FreqDist
from nltk import regexp_tokenize
from nltk.stem import WordNetLemmatizer

#----------------------   Util Functions    ------------------------
# ------------------------------------------------------------------

def corpus_list_to_array(corpus):
    #assert len(corpus[1]) == len(corpus[5])
    token_sentence = len(corpus)
    word = len(corpus[0])
    corpus_array = np.zeros(shape = (token_sentence,word), dtype= 'int')
    #print(corpus_array.shape, token_sentence, word)
    for s in range(token_sentence):
        corpus_array[s, :] = np.array(corpus[s], dtype='int')

    return corpus_array


#------------------------   Preprocess and Tokenize Steps/Layers    ------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
class Lexikos:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab = set()
        self.vocab_size = 0
        self.sentence_length = 0

        self.token_dim = 0
        self.vectors = np.array([])


        #funcs to add to this obj
        #token_dim = 0
        #vectors = np.array([])

        #v2i =
        #v2w
        #i2v
        #w2v


        pass


    # -------------------------------------------------------

    def set_token_step(self, tokenizer_object):
        '''
        :param tokenizer_object: instance of the Tokenizer class object that has been run on a corpus of text
            pulls from the Tokenizer object
        '''

        self.word_to_index, self.index_to_word = tokenizer_object.get_luts()
        self.vocab, self.vocab_size = tokenizer_object.get_vocab()
        self.sentence_length = tokenizer_object.sentence_length


    def set_vector_step(self, vectorizer_object):
        '''
        :param vectorizer_object: can be either w2vec, GLOVE, or BlindEmbed
        :return:
        '''
        self.token_dim = vectorizer_object.dim

        self.vectors = copy.copy(vectorizer_object.weights)
        # v2i
        # v2w
        # i2v
        # w2v

    def set_positional_step(self, positional_object):
        pass

    # -------------------------------------------------------

    def get_tokenizer_data(self):
        return self.vocab, self.vocab_size, self.sentence_length, self.index_to_word, self.word_to_index

    def get_vector_data(self):
        return self.vocab, self.vocab_size, self.token_dim, copy.copy(self.vectors), self.index_to_word, \
               self.word_to_index



    # -------------------------------------------------------

    @classmethod
    def load(cls, filename):
        '''
        object = Lexikos.load('filename.lxk')
        :param filename: filename in cwd for .lxk file
        :return:
        '''
        with open(filename, 'rb') as handle:
            return pickle.load(handle)


    def save(self, filename):
        '''
        :param filename: filename to dump pickled file into - should end in .lxk
        '''
        assert type(filename) == str

        if '.lxk' not in filename:
            filename += '.lxk'

        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

        print(f'saved Lexikos obj to {filename}')


# --------------------------- T O K E N I Z E R -------------------------------------------------
# ------------------------------------------------------------------------------------------------

class Tokenizer:
    def __init__(self):
        '''
        :param lexikos_object: default to None if not continuing to add to existing vocabulary and LUTS; set to a
        Lexikos object otherwise, to hold LUTS for different NLP stages, tokenizer, vectorizer and positional encoding.
        '''
        self.stopwords = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by" "can't",
             "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
             "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
             "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
             "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
             "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
             "through", "to", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
             "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
             "you", "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}
        #self.stopwords = {"a", "the", "to"}

        self.regex_pattern = r"([a-zA-z'-a-zA-Z]*)|\w+|\...|\..|\.|\!|\?|\-"
        self.start_token = '_START_'
        self.end_token = '_END_'
        self.pad_token = '_PAD_'
        self.unknown_token = '_UNK_'

        self.lemmatizer = WordNetLemmatizer() #Lemmatizer may be worth expanding in future

        self.vocab = set()
        self.vocab_size = 0
        self.sentence_length = 0
        self.index_to_word = []
        self.word_to_index = []

        self.resume = False


    def tokenize_by_sent(self, filename, pad_to_length = 0, build_luts = True, lexikos_object = None):
        '''Tokenizes a block of text (like project gutenberg - splitting by sentence
        :param filename: string, path to txt file of corpus to load
        :param pad_to_length: 0 or int, if int insert pad tokens up to the maximum length specified by
                self.sentence_length
        :param build_luts: Boolean, True to create word_to_index and index_to_word from this text
        :param lexikos_object: Lexikos object containing already existing vocab and LUT objects
        :return: returns the tokenized, lemmatized and cleaned corpus
        '''
        self.regex_pattern = r"([a-zA-z'-a-zA-Z]*)|\w+|\...|\..|\.|\!|\?|\-"

        self.sentence_length = pad_to_length
        tokenized_block = self.load_file(filename)

        if lexikos_object != None:
            #using Lexikon object, add to existing vocab
            self.vocab, self.vocab_size, self.sentence_length, self.index_to_word, \
            self.word_to_index = lexikos_object.get_tokenizer_data()
            self.resume = True
        else:
            self.resume = False

        tokenized_sentences = []

        if tokenized_block != []:
            sentence_start = 0
            sentence_end = 0
            for w_i, word in enumerate(tokenized_block):
                word = self.lemmatizer.lemmatize(word)
                word = word.replace(',', '').replace(';', '').replace(':', '')

                # trying stopwords first!
                if word in self.stopwords:
                    word = ''
                    tokenized_block[w_i] = word

                #may be unnecessary with full stopwords removed
                elif len(word) < 2 and word != '.': #if len(word) < 2 and word != '.':
                    word = ''
                    tokenized_block[w_i] = word

                elif word[-1] == '.':
                    tokenized_block[w_i] = word
                    sentence_end = w_i
                    sent = tokenized_block[sentence_start: sentence_end]
                    while sent.count(''):
                        sent.remove('')

                    if sent == []:
                        pass
                    else:
                        sent.insert(0, self.start_token)  # inserting sentence flags for start
                        sent.append(self.end_token)
                        while len(sent) < pad_to_length:
                            sent.append(self.pad_token)
                        sent.append(self.end_token) # inserting sentence flags for stop
                        sentence_start = w_i + 1
                        tokenized_sentences.append(sent)

                else:
                    tokenized_block[w_i] = word
        else:
            raise NameError('file empty')
        indexed_sentences = []
        #print(f'after tokenize{tokenized_sentences[:10]}')


        if self.resume:
            word_freq = FreqDist(itertools.chain(*tokenized_sentences))
            local_vocab = set(word_freq.most_common())
            full_vocab = self.vocab.union(local_vocab)
            self.vocab_size = len(full_vocab)
            print(f"Continuing to munch words, found {self.vocab_size} uniques")

            for word, w_i in local_vocab:
                if word not in self.vocab:
                    self.index_to_word.append(word)
                    self.vocab.add(word)

            self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
            print(f'Continuing tokenize, {len(tokenized_sentences)} sentences \nMade up of {self.vocab_size} uniques')

        elif build_luts: #if we are building out LookUpTables for the first time...
            word_freq = FreqDist(itertools.chain(*tokenized_sentences))
            local_vocab = set(word_freq.most_common())
            self.vocab_size += len(local_vocab) + 4 #adding 4 for tokens
            print(f"Found {self.vocab_size} unique words")


            if self.index_to_word == []:
                tokens = [self.start_token, self.end_token, self.pad_token, self.unknown_token]
                for tok in tokens:
                    self.index_to_word.append(tok)

            for word, w_i in local_vocab:
                if word not in self.vocab:
                    self.index_to_word.append(word)
                    self.vocab.add(word)

            self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
            print(f'tokenized {len(tokenized_sentences)} sentences \nMade up of {self.vocab_size} uniques')

        #for unknown words - replace unknowns with the unk token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [word if word in self.word_to_index else self.unknown_token for word in sent]
            #tokenized_sentences[i] = [w if w in self.word_to_index else self.unknown_token for w in sent]
            indexed_sentences.append([self.word_to_index[word] if word in self.word_to_index else
                                      self.word_to_index[self.unknown_token] for word in sent])

        return tokenized_sentences, indexed_sentences


    def tokenize_by_chunk(self, chunk_data, pad_to_length = 0, build_luts = True, lexikos_object = None):
        '''Tokenizes by review or iterable block rather than by splitting into sentences
        :param filename: string, path to txt file of corpus to load
        :param pad_to_length: 0 or int, if int insert pad tokens up to the maximum length specified by
                self.sentence_length
        :param build_luts: Boolean, True to create word_to_index and index_to_word from this text
        :param lexikos_object: Lexikos object containing already existing vocab and LUT objects
        :return: returns the tokenized, lemmatized and cleaned corpus
        '''
        self.regex_pattern = r"([a-z'0-9A-Z]*)"

        self.sentence_length = pad_to_length if pad_to_length > 0 else self.sentence_length


        if lexikos_object != None:
            #using Lexikon object, add to existing vocab
            self.vocab, self.vocab_size, self.sentence_length, self.index_to_word, \
            self.word_to_index = lexikos_object.get_tokenizer_data()
            self.resume = True
        else:
            self.resume = False

        tokenized_final = []
        for tokenized_block in chunk_data:
            if tokenized_block != []:
                length = len(tokenized_block)
                for w_i, word in enumerate(tokenized_block):
                    word = self.lemmatizer.lemmatize(word)
                    word = word.replace("'", "")

                    if word in self.stopwords:
                        word = ''
                        tokenized_block[w_i] = word
                    else:
                        tokenized_block[w_i] = word

                    #if we reach the end of the chunk
                    if w_i + 1 == length:
                        #while there are empty tokens, take them out until they're gone!
                        while tokenized_block.count('') > 0:
                            try:
                                tokenized_block.remove('')
                            except ValueError:
                                break

                        # if tokenized_block is empty (all words removed or other error), break out
                        if tokenized_block == []:
                            break

                        #if there is content in tokenized_block and we're at the end...
                        else:
                            tokenized_block.insert(0, self.start_token)
                            tokenized_block.append(self.end_token)

                            #if we are padding to a specific length, append our pad_token
                            while len(tokenized_block) < self.sentence_length:
                                tokenized_block.append(self.pad_token)

                        #append this block to our processed list of lists
                        tokenized_final.append(tokenized_block)

            else:
                raise NameError('file empty')
        indexed_sentences = []

        if self.resume:
            word_freq = FreqDist(itertools.chain(*tokenized_final))
            local_vocab = set(word_freq.most_common())
            full_vocab = self.vocab.union(local_vocab)
            self.vocab_size = len(full_vocab)
            print(f"Continuing to munch words, found {self.vocab_size} uniques")

            for word, w_i in local_vocab:
                if word not in self.vocab:
                    self.index_to_word.append(word)
                    self.vocab.add(word)

            self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
            print(f'Continuing tokenize, {len(tokenized_final)} sentences \nMade up of {self.vocab_size} uniques')

        elif build_luts: #if we are building out LookUpTables for the first time...
            word_freq = FreqDist(itertools.chain(*tokenized_final))
            local_vocab = set(word_freq.most_common())
            self.vocab_size += len(local_vocab) + 4 #adding 4 for tokens
            print(f"Found {self.vocab_size} unique words")

            if self.index_to_word == []:
                tokens = [self.start_token, self.end_token, self.pad_token, self.unknown_token]
                for tok in tokens:
                    self.index_to_word.append(tok)

            for word, w_i in local_vocab:
                if word not in self.vocab:
                    self.index_to_word.append(word)
                    self.vocab.add(word)

            self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
            print(f'tokenized {len(tokenized_final)} sentences \nMade up of {self.vocab_size} uniques')

        #for unknown words - replace unknowns with the unk token
        for i, sent in enumerate(tokenized_final):
            tokenized_final[i] = [word if word in self.word_to_index else self.unknown_token for word in sent]
            #tokenized_final[i] = [w if w in self.word_to_index else self.unknown_token for w in sent]
            indexed_sentences.append([self.word_to_index[word] if word in self.word_to_index else
                                      self.word_to_index[self.unknown_token] for word in sent])

        return tokenized_final, indexed_sentences


    def load_file(self, filename):
        if os.path.exists(filename):
            print(f'loading {filename}')
            text = open(filename, 'r').read()
            while '\n' in text:
                text = text.replace('\n', ' ')
            while '?' in text:
                text = text.replace('?', '.')
            while '!' in text:
                text = text.replace('!', '.')
            while '_' in text:
                text = text.replace('_', ' ')
            while '-' in text:
                text = text.replace('-', ' ')

            return regexp_tokenize(text = text.lower(),
                                   pattern = self.regex_pattern,
                                   gaps = False,
                                   discard_empty = True)
        else:
            raise FileNotFoundError('check the filename')


    def load_csv(self, filename):
        if os.path.exists(filename):
            print(f'loading {filename}')
            #np.read_csv...########TODO: read csv in; format to work with self.tokenize_chunk()
            text = open(filename, 'r').read()
            while '\n' in text:
                text = text.replace('\n', ' ')
            while '?' in text:
                text = text.replace('?', '.')
            while '!' in text:
                text = text.replace('!', '.')
            while '_' in text:
                text = text.replace('_', ' ')
            while '-' in text:
                text = text.replace('-', ' ')

            return regexp_tokenize(text = text.lower(),
                                   pattern = self.regex_pattern,
                                   gaps = False,
                                   discard_empty = True)
        else:
            raise FileNotFoundError('check the filename')

    def find_longest_sent(self, corpus):
        '''
        :param
        '''
        length = [len(sen) for sen in corpus]
        self.sentence_length = max(length) + 2 #adding two for start and stop tokens
        print('max L is', self.sentence_length)
        return self.sentence_length

    def pad_to_length(self, sentence):
        pass

    def get_luts(self):
        '''
        :return: returns tuple of word_to_index LUT and index_to_word LUT
        '''
        return (self.word_to_index, self.index_to_word)

    def get_vocab(self):
        '''
        :return: returns our vocab set, and our vocab count
        '''
        return self.vocab, self.vocab_size

    def __str__(self):
        return f'vocab of {self.vocab_size}, '


if __name__ == '__main__':

    tokenizer = Tokenizer()
    corpus, indexed_corpus = tokenizer.tokenize_by_sent(filename = 'doriangray.txt', pad_to_length = 0,
                                                        build_luts = True)

    #corpus, indexed_corpus = tokenizer.tokenize_by_chunk(filename='imdb_train.csv', pad_to_length=0,
    #                                                    build_luts=True)
    w2i, i2w = tokenizer.get_luts()
    vocab, vocab_size = tokenizer.get_vocab()
    print(vocab_size, len(i2w))
    print(corpus[:5])


    max_l = 0
    max_i = 0
    for s_i, sent in enumerate(corpus):
        this_l = len(sent)
        if this_l > max_l:
            max_l = this_l
            max_i = s_i
    print(f'Max S length {max_l}')
    print(corpus[max_i])

    #array_corp = corpus_list_to_array(indexed_corpus)

    #pandasdf
    #pandasdf['length'] = pandasdf['text'].apply(len)