#---------------------------Main Classifier File--------------------------

class ImageClassifier(object):

    def __init__(self):
        self.name = 'Classifier'
        #----------------data->data_class->data_set->feature->data----------------
        self.data = {'digit':{  'train':{   'image_file':{'path':'data/digitdata/trainingimages'},
                                            'label_file':{'path':'data/digitdata/traininglabels'}},
                                'test':{    'image_file':{'path':'data/digitdata/testimages'},
                                            'label_file':{'path':'data/digitdata/testlabels'}},
                                'valid':{   'image_file':{'path':'data/digitdata/validationimages'},
                                            'label_file':{'path':'data/digitdata/traininglabels'}}},
                    'face':{    'train':{   'image_file':{'path':'data/facedata/facedatatrain'},
                                            'label_file':{'path':'data/facedata/facedatatrainlabels'}},
                                'test':{    'image_file':{'path':'data/facedata/facedatatest'},
                                            'label_file':{'path':'data/facedata/facedatatestlabels'}},
                                'valid':{   'image_file':{'path':'data/facedata/facedatavalidation'},
                                            'label_file':{'path':'data/facedata/facedatavalidationlabels'}}}}

        #--------------------LOADING DATA INTO THE DICTIOANRY-------------------------
        for data_class in self.data.values():
            for data_set in data_class.values():
                labels = []
                images = []
                labels_file =  open(data_set['label_file']['path']).readlines()
                image_file = open(data_set['image_file']['path']).readlines()
                for line in labels_file:
                        if not line.isspace(): labels.append(int(line))
                image_height = len(image_file)/len(labels)
                for i in range(0,len(labels)): images.append(image_file[i*image_height:(i+1)*image_height])
                data_set['label_file']['data'] = labels
                data_set['image_file']['data'] = images
        #-------------------FORMATING DATA FOR ALGO----------------------
        for data_class in self.data.values():
            for data_set in data_class.values():
                images = data_set['image_file']['data']
                labels = data_set['label_file']['data']
                classifications = []
                features = []
                for label in labels:
                    encoded_classification = []
                    encoded_classification.append(label)
                    classifications.append(encoded_classification)
                for image in images:
                    encoded_feature = []
                    for line in image:
                        for char in line:
                            encoded_feature.append(ord(char))
                    features.append(encoded_feature)
                data_set['classification'] = classifications
                data_set['features'] = features
        #-------------------END OF CLASSIFIER----------------------


def main():
    classifier = ImageClassifier()
    print "Welcome"
    #print classifier.data['digit']['test']['features'][0:2]
    print [len(f) for f in classifier.data['digit']['test']['features'][0:1000:10]]



if __name__ == "__main__":
    main()









"""     rough draft




#----------------------LOADING LABELS INTO ARRAY--------------------------
def load_labels_into_array():
    labels = []
    for line in open(data['digit']['train']['label_file']['path']).readlines():
        if not line.isspace():
            labels.append(int(line))

#-------------------BREAKS UP IMAGE BY WHITESPACE--------------------------
def break_up_image_by_whitespace():
    images = []
    current = []
    for line in open(data['digit']['train']['image_file']['path']).readlines():
        if not line.isspace():
            current.append(line)
        if current and line.isspace():
            images.append(current)
            current = []

#--------------------CALCULATING LINES IN AN IMAGE----------------------
def calculate_image_height():
    heights =
    for current in image:
         if len(current) not in heights: heights[len(current)] = []
         heights[len(current)].append(image.index(current))
#---------------PRINTING LENGTH OF DICTIONARY-------------------------
def print_length_of_dictionary():
    for data_class in data:
            for data_set in data[data_class]:
                    for feature in data[data_class][data_set]:
                            print data_class, data_set, feature, len(data[data_class][data_set][feature]['data'])
    for line in data[data_class][data_set][feature][data][index]: print line
#------------------------------------------------------------------------
"""
