from tensorflow.keras.preprocessing.sequence import pad_sequences
from unicodedata import normalize

class StringVectorizer:
    """A set of functions to encode strings into integer arrays given a character set.
    
    Parameters
    ----------
        max_word_length: int, Default 34
            Set by the most common longest word in French, English and Spanish
            Sources:
                https://en.wikipedia.org/wiki/Longest_word_in_Spanish
                https://en.wikipedia.org/wiki/Longest_word_in_English
                https://en.wikipedia.org/wiki/Longest_word_in_French
    """
    # CHARSET = WordHTRFlor.LATIN_CHAR
    CHARSET = " !\"#$%&'()*+,-.0123456789:;<>@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáÁéÉíÍóÓúÚëËïÏüÜñÑçÇâÂêÊîÎôÔûÛàÀèÈùÙ"
    
    def __init__(self, max_word_length=34):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + StringVectorizer.CHARSET)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_word_length
    
    def vectorize(self, word, as_onehot=False):
        """Encode word to integer vector (or binary vector if 'as_onehot' set to True)
        Spanish characters work with NFKC, not NFKD (https://docs.python.org/3/library/unicodedata.html)

        Source:
            https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/data/generator.py#L115
        """
        word = normalize("NFKC", word).encode("UTF-8", "ignore").decode("UTF-8")
        
        vector = (self.chars.find(word[i]) for i in range(len(word)))
        vector = [num if num != -1 else self.UNK for num in vector]
                
        padded_vector = pad_sequences([vector], maxlen=self.maxlen, padding="post")
        return padded_vector[0]
        
    def decode(self, vector, as_onehot=False):
        """Inverses the vectorization"""
        chars = "".join([self.chars[pos] for pos in vector])
        return chars.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
    