#---------------------------Main Classifier File--------------------------

class ImageClassifier(object):

    def initData(self):
        self.data = {
            'digit':{
                'train':{
                    'image_file':{'path':'data/digitdata/trainingimages'},
                    'label_file':{'path':'data/digitdata/traininglabels'}},
                'test':{
                    'image_file':{ 'path':'data/digitdata/testimages'},
                    'label_file':{'path':'data/digitdata/testlabels'}},
                'valid':{
                    'image_file':{'path':'data/digitdata/validationimages'},
                    'label_file':{'path':'data/digitdata/traininglabels'}}},
            'face':{
                'train':{
                    'image_file':{'path':'data/facedata/facedatatrain'},
                    'label_file':{'path':'data/facedata/facedatatrainlabels'}},
                'test':{
                    'image_file':{'path':'data/facedata/facedatatest'},
                    'label_file':{'path':'data/facedata/facedatatestlabels'}},
                'valid':{
                    'image_file':{'path':'data/facedata/facedatavalidation'},
                    'label_file':{'path':'data/facedata/facedatavalidationlabels'}}}}

    def loadData(self):
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


    def formatData(self):
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


    def printStructure(self, structure, depth):
        ret = ""
        limit = 5
        if isinstance(structure, type('a')):
            ret += ('\t'*depth) + (structure) + ('\n')
        if isinstance(structure, type(1)):
            ret += ('\t'*depth) + (str(structure))[:limit] + ('\n')
        if isinstance(structure, type((1,2))):
            ret += ('\t'*depth) + str(structure[0]) + (": ") + self.printStructure(structure[1], depth)[depth:]
        if isinstance(structure, type([])):
            ret += ('\t'*depth) + ('[') + ('\n')
            for i in range(0,len(structure))[:limit]: ret += self.printStructure((i,structure[i]), depth+1)
            ret = ret[:len(ret)-1] + (']') + ('\n')
        if isinstance(structure, type({})) and len(structure):
            ret += ('\t'*depth) + ('{') + ('\n')
            for (k,v) in structure.iteritems(): ret += self.printStructure((k,v), depth+1)
            ret = ret[:len(ret)-1] + ('}') + ('\n')
        return ret

    def printStruct(self, structure):
        print self.printStructure(structure, 0)


    def __init__(self):
        self.initData()
        self.loadData()
        self.formatData()

#-------------------END OF CLASSIFIER----------------------
#from ImageClassifier import ImageClassifier
def main():
    print "Welcome"
    #classifier = ImageClassifier()
    #classifier.printStruct(classifier.data)
    #classifier.loadData()
    #classifier.printStruct(classifier.data)
    #classifier.formatData()
    #classifier.printStruct(classifier.data)
    #classifier.printStruct(classifier.data['face']['train']['image_file'])
    #print classifier.data['digit']['test']['features'][0:2]
    #print [len(f) for f in classifier.data['digit']['test']['features'][0:1000:10]]

if __name__ == "__main__":
    main()
