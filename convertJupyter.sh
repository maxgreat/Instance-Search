#!/bin/bash
jupyter nbconvert --to python model/ModelDefinition.ipynb
jupyter nbconvert --to python dataset/collection.ipynb
jupyter nbconvert --to python dataset/ReadImages.ipynb
jupyter nbconvert --to python TrainClassif.ipynb
jupyter nbconvert --to python model/siamese.ipynb

