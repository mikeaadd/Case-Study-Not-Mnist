### Case Study: not-MNIST images

__data:__ 28x28 pixel images of letters in all variations of font

### steps:
__split data__ in to train and validation sets (80/20)

__basic model:__ get a basic KERAS sequential CNN model to run
 - convert to greyscale for computational performance (already b/w)
 -  no image transformations ... they are already hard enough
 - initial performance: 75% accuracy on unseen test data
 - curious: validation accuracy > training accuracy

![initial results](/src/figs/test2_accuracy_curves.png)

__optimize model:__

 | epochs    | batch size | optimization |  activation | dropout |  training accuracy | test accuracy |
 |----------|----------|---------|--------|------|----|----|
 | 15 | 200 | sgd | RelU | 0.5 | 0.661 |
 | 15 | 200 | adadelta | ReLU |0.5 | 0.787 |
 | 15 | 200 | adagrad | ReLU | 0.5 |0.786 |
 | 15 | 200 | adam | ReLU | 0.5 |0.791  |
 | 15 | 200 | nadam |  ReLU |0.5 | 0.807 |
 | 15 | 200 | nadam | sigmoid | 0.5 |0.642 |
 | 30 | 200 | nadam | ReLU | 0.1 |0.755 | 0.725 |
 | 30 | 200 | nadam | ReLU | 0.1 |0.761 | 0.779 |
 | 30 | 200 | nadam | ReLU | 0.9 |0.880 | 0.879 |


 best Conv2D params: kernal:(3,3), strides:(1,1), padding:'same'
