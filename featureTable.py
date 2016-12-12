
import numpy

numDigs = 10
rows = 20;#more for facial
cols = 30;

data = open('data/digitdata/trainingimages')

#must seperate digits here and store them in their individual arrays

# for array in arrays:
    # digit = what this array (image) corresponds to (known because this is training)
    # call table_builder_digit(array) to update frequency table

digit = 0; # this needs to be whatever the digit actually is

# 1 row per feature (pixel at (x, y))
# yes/no for each digit
feature_prob_table = numpy.zeros((2*rows*cols, 20));

# input: array = [0, 0, 0, ..., 1, 0, 0, 1, 1, ...]
# output: void, feature_prob_table is loaded with values
def table_builder_digit(array):
    for i in range(0, len(array)):
        if array[i] == 1:
            row = 2 * i
        else:
            row = 2 * i + 1
        for col in range(1, (2*numDigs), 2):# assume none is digit
            print col
            if int(col / 2) is digit:
                feature_prob_table[row, col-1] = feature_prob_table[row, col-1] + 1;
            else:
                feature_prob_table[row, col] = feature_prob_table[row, col] + 1;

# store this table to file

a = numpy.array([1]*600)
numpy.random.shuffle(a)
table_builder_digit(a)
print feature_prob_table
