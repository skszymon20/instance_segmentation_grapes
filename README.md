# instance_segmentation_grapes
Instance segmentation on wgisd dataset for Computer Vision course on PP.
Inspirations for the project came from the paper: https://arxiv.org/pdf/1907.11819.pdf

## Tags
To view repository status from the moment of the Report.pdf please view tag v0.0.1

## Our setup:
 - First clone wgisd git repository: $git clone https://github.com/thsant/wgisd.git. It should be in current working directory.
 - Install libraries which are listed in "requirements.txt". For example You can use $pip install -r requirements.txt
 - We used python 3.9.0 for running experiments.

## Get started
Files with prefixes evaluate are used to evaluate models. Files starting with prefix run are used to train the model. We performed 3 experiments:
 - experiment1: (evaluate.ipynb and runMaskRCNN4lrscheduled) runs and evaluates MaskRCNN model. We used scheduled learning rate (StepLR) and binary_cross_entropy_with_logits as a loss function. We plotted loss function during training on both validataion and training dataset. Moreover we plotted both total loss and mask loss. Then, we evaluated final results on validation dataset using precision, recall and fscore. 
 - experiment2: (evaluate_losses.ipynb and runMaskRCNN5differentlossfoos.ipynb) runs MaskRCNN using 3 different loss functions: Dice, IoU(Intersection over Union) and binary_cross_entropy_with_logits. Final results are evaluated using the same metrics as in the previous point.
 - experiment3: (evaluate_losses_imaug (1).ipynb, runMaskRCNN6imgaugs.ipynb) contains similar run and evaluation as in the first point. We selected loss function to be binary_cross_entropy_with_logits. Then, we performd image augmentations using albumenatations library. Final results are evaluated using precision, recall and fscore.
 - plot_samples.ipynb is a notebook that contains visualisation of sample predictions of our final model. The final model is the model obtained in experiment3
 
## Credit
 - We thank to https://github.com/thsant/wgisd for sumplementing us with the dataset and inspirations.
 - We thank kaggle's user bigironshere for posting survey on loss functions for Instance Segmentation: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

## Authors:
 - Szymon Skwarek
 - Magdalena Kobusińska

## Contact:
 - email: skszymon20@gmail.com
