# instance_segmentation_grapes
Instance segmentation on wgisd dataset for Computer Vision course on PP.
We thank to https://github.com/thsant/wgisd for sumplementing us with the dataset.
Project Idea comes from the paper: https://arxiv.org/pdf/1907.11819.pdf

## Our setup:
First clone wgisd git repository: $git clone https://github.com/thsant/wgisd.git
We used python 3.9.0 for running experiments.
Install libraries which are listed in "requirements.txt"
For example You can use $pip install -r requirements.txt

## Get started
Project consists of 4 notebooks: evaluate.ipynb, evaluate_losses.ipynb, evaluate_losses_imaug (1).ipynb, plot_samples.ipynb. Notebooks listed here are sorted from the oldest to the latest. 
 - evaluate.ipynb evaluates MaskRCNN model on validation dataset. We used scheduled learning rate (StepLR) and binary_cross_entropy_with_logits as a loss function. We plotted loss function during training on both validataion and training dataset. Moreover we plotted both total loss and mask loss. Then, we evaluated final results on validation dataset using precision, recall and fscore.
 - evaluate_losses.ipynb contains experiments from evaluate run using 3 different loss functions: Dice, IoU(Intersection over Union) and binary_cross_entropy_with_logits. Final results are evaluated using the same metrics as in the previous point.
 - evaluate_losses_imaug (1).ipynb contains similar evaluation as in the first point. We selected loss function to be binary_cross_entropy_with_logits. Then, we performd image augmentations using albumenatations library. Final results are evaluated using precision, recall and fscore.
 - plot_samples is a notebook that contains visualisation of sample predictions of our final model. The final model is the model obtained via evaluate_losses_imaug (1).ipynb
 
## Authors:
 - Szymon Skwarek
 - Magdalena Kobusińska

## Contact:
 - mail: skszymon20@gmail.com
