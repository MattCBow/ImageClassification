#---------------------------Main Classifier File--------------------------
#problem->type->feature->
data = {'digit':{   'train':{   'images':{'file_path':'data/digitdata/trainingimages'},
                                'labels':{'file_path':'data/digitdata/traininglabels'}},
                    'test':{    'images':{'file_path':'data/digitdata/testimages'},
                                'labels':{'file_path':'data/digitdata/testlabels'}},
                    'valid':{   'images':{'file_path':'data/digitdata/validationimages'},
                                'labels':{'file_path':'data/digitdata/traininglabels'}}},
                    'face':{    'train':{'images':{'file_path':'data/facedata/facedatatrain'},
                                'labels':{'file_path':'data/facedata/facedatatrainlabels'}},
                    'test':{    'images':{'file_path':'data/facedata/facedatatest'},
                                'labels':{'file_path':'data/facedata/facedatatestlabels'}},
                    'valid':{   'images':{'file_path':'data/facedata/facedatavalidation'},
                                'labels':{'file_path':'data/facedata/facedatavalidationlabels'}}}}

#----------------------LOADING LABELS INTO ARRAY--------------------------
label = []
for line in open(data['digit']['train']['labels']['file_path']).readlines():
    label.append(int(line))
#----------------------LOADING IMAGES INTO ARRAY--------------------------
image = []
current = []
for line in open(data['digit']['train']['images']['file_path']).readlines():
    if not line.isspace():
        current.append(line)
    if current and line.isspace():
        image.append(current)
        current = []
#--------------------LOADING DATA INTO THE DICTIOANRY-------------------------

for data_problem in data:
    for data_type in data[data_problem]:
        data[data_problem][data_type]['labels']['data'] = []
        for line in open(data[data_problem][data_type]['labels']['file_path']).readlines():
            data[data_problem][data_type]['labels']['data'].append(int(line))
        current = []
        data[data_problem][data_type]['images']['data'] = []
        for line in open(data[data_problem][data_type]['images']['file_path']).readlines():
            if not line.isspace():
                current.append(line)
            if current and line.isspace():
                data[data_problem][data_type]['images']['data'].append(current)
                current = []


#--------------------PRINTING LINES IN AN IMAGE----------------------
heights = {}
for current in image:
     if len(current) not in heights: heights[len(current)] = []
     heights[len(current)].append(image.index(current))
#---------------PRINTING LENGTH OF DICTIONARY-------------------------
for data_problem in data:
        for data_type in data[data_problem]:
                for feature in data[data_problem][data_type]:
                        print data_problem, data_type, feature, len(data[data_problem][data_type][feature]['data'])

#------------------------------------------------------------------------
