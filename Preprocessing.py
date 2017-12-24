import os
from collections import Counter
import numpy as np

class Preprocessing:

    def __init__(self):
        self.__train_dir = None
        self.__test_dir = None

    def set_directory(self, dir):
        self.__train_dir = dir


    def get_emails(self):
        """
        :param train_dir: local directory that contains text files of the emails
        :return: list of all files (full directory) for every email
        """
        emails = [os.path.join(self.__train_dir, f) for f in os.listdir(self.__train_dir)]
        return emails

    def get_emails_size(self):
        return len(self.get_emails())

    def words_count(self):
        """
        :return: dictionary represents counts of all words in every email
        """
        counts = {}
        # for every email:
        for mail in self.get_emails():
            # open the file
            with open(mail) as m:
                for i, line in enumerate(m):
                    # email body in the third line:
                    if i == 2:
                        # get all words and count them:
                        words = line.split()
                        for word in words: counts[word] = counts.get(word, 0) + 1
        return self.__clean_dict(Counter(counts))

    def __clean_dict(self, dict_counts):
        """
        responsible for removing non-words and absured single characters which are irrelevant
        :param dict_counts: dictionary contains every words occurrences in every email
        :return: cleaned and filtered dictionary
        """
        to_be_removed = dict_counts.copy().keys()
        for item in to_be_removed:
            # delete all non-alphabetic entries:
            if not item.isalpha():
                del dict_counts[item]
            # delete all single letters:
            elif len(item) == 1:
                del dict_counts[item]
        return dict_counts

    def build_sparse_feat_matrix(self):
        """
        this function responsible for feature extraction process
        Feature space here is 4000 dimensions for each email [emails_size, features_size]
        :return: feature matrix that is bag of words model representation
        """
        features_n = len(self.words_count().most_common(1000))
        emails_n = len(self.get_emails())
        matrix = np.zeros((emails_n, features_n))
        doc_idx, word_idx = 0, 0

        for d, email in enumerate(self.get_emails()):
            # for every email/document:
            with open(email) as mail:
                # open the email file
                for i, line in enumerate(mail):
                    # get the email body only
                    if i == 2:
                        words = line.split()
                        for word in words:
                            word_idx = 0
                            for i, w in enumerate(self.words_count()):
                                if w == word:
                                    word_idx, doc_idx = i, d
                                    matrix[doc_idx, word_idx] = words.count(word)

        return matrix
