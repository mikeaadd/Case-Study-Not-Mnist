### Case Study: not-MNIST images

__data:__ 28x28 pixel images of letters in all variations of font

### steps:
split data in to train and validation sets (80/20)

get a basic KERAS sequential CNN model to run
 - convert to greyscale for computational performance (already b/w)
 -  no image transformations ... they are already hard enough
 - initial performance: 80% accuracy
 - curious: validation accuracy > training accuracy 

![initial results](/src/figs/test2_accuracy_curves.png)
