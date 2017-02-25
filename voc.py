import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)


class VOCDataset:

    def __init__(self, root, sets):
        self.root = root

        self.images = list()
        for year, name in sets:
            root = os.path.join(self.root, 'VOC' + year)
            for line in open(
                    os.path.join(root, 'ImageSets', 'Main', name + '.txt')):
                self.images.append((root, line.strip()))

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i][1]

    def image(self, i):
        return cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR)

    def annotation(self, i):
        boxes = list()
        classes = list()
        difficulties = list()
        tree = ET.parse(os.path.join(
            self.images[i][0], 'Annotations', self.images[i][1] + '.xml'))
        # print(self.images[i][0], 'Annotations', self.images[i][1] + '.xml')))
        root = tree.getroot()
        for object_tree in root.findall('object'):
            bndbox = object_tree.find('bndbox')
            boxes.append(tupletuple(float(bndbox.find(t).text)
                                    for t in ('xmin', 'ymin', 'xmax', 'ymax')))
            try:
                classes.append(self.names.index(object_tree.find('name').text))
                # print(object_tree.find('name').text) 
                difficulties.append(bool(int(child.find('difficult').text)))
            except:
                pass                   
        boxes = np.array(boxes)
        classes = np.array(classes)
        difficulties = np.array(difficulties)
        return boxes, classes, difficulties
