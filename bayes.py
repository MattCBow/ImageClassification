    # bayes classifier
    # input: array of pixels, 0's and 1's, representing a digit
import numpy
import math
from ImageClassifier import ImageClassifier

class LearnBayes():
    global images
    global image
    global feature_prob_table
    global numDigs
    numDigs = 10
    global rows
    rows = 20;#more for facial
    global cols
    cols = 30;
    global freq_map
    freq_map = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    global digit

    #------------------COMPUTE P(D = d | E)-----------------------

    def bayes(self, digitm, imagem):
        res = 1
        for i in range(len(imagem)):
            if imagem[i] == (35 or 43):
                row = 2 * i + 1# 0 comes first in frequency table
            else:
                row = 2 * i
            if numpy.any(feature_prob_table[row, (digitm*2)] > 2):# ensure no 0 multiplication
                res = res * (feature_prob_table[row, digitm*2] / freq_map[digitm])
        return res

    # input: array = [0, 0, 0, ..., 1, 0, 0, 1, 1, ...]
    # output: void, feature_prob_table is loaded with values
    def table_builder_digit(self):
        global freq_map
        freq_map[digit] += 1
        for i in range(0, len(image)):
            if image[i] == (35 or 43):
                row = 2 * i + 1# 0 comes first in frequency table
            else:
                row = 2 * i
            for col in range(1, (2*numDigs), 2):# assume none is digit
                if int(math.floor(col / 2)) == digit:
                    feature_prob_table[row, col-1] = feature_prob_table[row, col-1] + 1;
                else:
                    feature_prob_table[row, col] = feature_prob_table[row, col] + 1

    def __init__(self, array_of_imgarrays, array_of_corrdigs):
        rows = 20;#more for facial
        cols = 30;
        global images, digits, feature_prob_table
        images = array_of_imgarrays
        digits = array_of_corrdigs
        feature_prob_table = numpy.zeros((2000, 20))
        for i in range(0, 50):
            # set values that get used by above functions
            global image, digit
            image = images[i]
            digit = digits[i][0]
            self.table_builder_digit()
        test_images = classifier.data['digit']['test']['features']
        test_digits = classifier.data['digit']['test']['classification']
        for i in range(len(test_images)):
            for j in range(0, 10):
                self.compP(j, test_images[i])

    def main():
