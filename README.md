# Pytorch models definition and test for image retrieval

This is the implementation of the paper :

Portaz, M., Kohl, M., Chevallet, J. P., Quénot, G., & Mulhem, P. (2019). Object instance identification with fully convolutional networks. Multimedia Tools and Applications, 78(3), 2747-2764.

If you use it, please cite :

	@article{portaz2019object,
		title={Object instance identification with fully convolutional networks},
		author={Portaz, Maxime and Kohl, Matthias and Chevallet, Jean-Pierre and Qu{\'e}not, Georges and Mulhem, Philippe},
		journal={Multimedia Tools and Applications},
		volume={78},
		number={3},
		pages={2747--2764},
		year={2019},
		publisher={Springer}
	}

## Test several approaches for images retrieval:
	* Feature Extraction from Pretrained CNN
	* Pretrained CNN finetuning
	* Siamese network from scratch
	* Siamese network with pretrained network

# TrainClassif
	Finetune a CNN for classification over few examples
	Finetune only the classifier or the entire network

# TrainSiamese
	Train a siamese network with pairs selection for image retrieval
