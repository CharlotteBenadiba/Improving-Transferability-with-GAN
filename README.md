# Improving-Transferability-with-GAN

This project focuses on enhancing the transferability of adversarial attacks across neural network models. It involves defining and training target models, implementing an AdvGAN attack with input diversity to improve attack transferability, evaluating the performance of the attacks on different target models, visualizing the results, and discussing the impact of data transformation on the success rate of the adversarial attacks.

## Table of Contents

- [Definitions of Neural Networks](#Definitions-of-Neural-Networks)
- [Training Target Models](#Training-Target-Models)
- [Adv_GAN Attack with input diversity](#Adv_GAN-Attack-with-input-diversity)
- [Attacking Target Models](#Attacking-Target-Models)
- [Test the performance of Adv_GAN](#Test-the-performance-of-Adv_GAN)
- [Visualization of the results](#Visualization-of-the-results)
- [Discussion](#Discussion)

## Files

* MNIST dataset available using Torch
* paper available at https://fr.overleaf.com/read/fbxbtvdvfgpy

----
## Definitions of Neural Networks

In this section we are first going to define the target models. There are three target models used in this project including two CNNs and a pre-trained ResNet50. The architecture of the two CNNs are as the followings:

    Conv-Relu -> Conv-Relu -> Maxpool -> Conv_Relu -> Conv_Relu -> Maxpool -> FullyConnectedLayer-Relu -> dropout -> FullyConnectedLayer_output_Relu

    Conv-Relu -> Conv-Relu -> Maxpool -> dropout -> flatten -> FullyConnetedLayer-Relu -> dropout -> FullyConnectedLayer_output_Softmax
    
Now,  we are going to build the Discriminator and the Generator in the Generative Adversarial NeuralNetwork. The architecture of the Disctiminator is : Conv -> LeakyRelu -> Conv ->BatchNorm -> LeakyRelu -> Conv-> BatchNorm-> LeakyRelu -> Conv -Sigmoid

----
## Training Target Models

This section is focused on defining hyperparameters and configuring the training process for a neural network model on the MNIST dataset. It includes setting up CUDA usage if available, loading the MNIST dataset, training a target model using a specified architecture, optimizing it with the Adam optimizer, adjusting learning rates during training, saving the trained model, and evaluating its accuracy on the MNIST test dataset.

----
## Adv_GAN Attack with input diversity

In this section, we are going to build the AdvGan with input diversity. The input diversity is an idea from Improving Transferability of Adversarial Examples with Input Diversity, Xie et al. 2018, in which they consider random resizing and rescaling to improve transferability. The general idea is more or less the same thing. Therefore we apply input diversity to AdvGAN to improve transferability.

----
## Attacking Target Models

During the training advGAN for attacking the target models, we notice that the loss of the generator of fake examples is increasing while the loss discriminator is decreasing. The perturbation loss nearly stays the same through all iterations. Finally the loss of advGAN network is decreasing and stablizing around 30 rounds.

In terms of the loss_G_fake, it is actually the discriminator loss when dealing with generated images. Therefore, it is supposed to go up, which means that the model successfully generate images that the discriminator cannot detect. 

----

## Test the performance of Adv_GAN

CUDA Available:  True
MNIST training dataset:
num_correct:  3609
accuracy of adv imgs in training set: 0.060150
train success rate:0.939850

num_correct:  579
accuracy of adv imgs in testing set: 0.057900
Test success rate:0.990350

CUDA Available:  True
MNIST training dataset:
num_correct:  16085
accuracy of adv imgs in training set: 0.268083
train success rate:0.731917

num_correct:  2825
accuracy of adv imgs in testing set: 0.282500
Test success rate:0.952917

The test results are shown as above. Due to the capacity, we only show an example of the transerability of the first target model. The adv_GAN in this scenario is that it was trained to attack the first model then we use this pretrained generator to attack the second target model. We could find that the success rate of both training set and the test sets are higher for Adv-GAN with data transformation. 

----
## Visualization of the results
Here is the display of a grid of images with their real and predicted labels
![Screenshot](/Images/original.png)

Here are the adversarial images generated using pretrained generator and data transformation
![Screenshot](/Images/generated1.png)

Here are the adversarial images generated using pretrained generator (no data transformation)
![Screenshot](/Images/generated2.png)

----
## Discussions

Based on the success rate, we could find that the data transformation actually helps to improve the transferability. We could also find the difference between the adversarial images with and without transformation.

However, the work is limited because we have only developed three target models. Two of them are convolutional neural networks, although the architectures are slightly different. It would be more convincing if we build several various target models with different structures to see if it actually works when transfer to other models. In the paper, the authors developed 8 CNN models as their target models. The results demonstrate that it had effectively prevents the generator from overfitting and improves transferability. However, from their results, the data enhancement methods does not work well for some of the target models, but the success rate of some models were really high, which means that this method might not generalize to all models. We may adjust transformations regarding different models to see if we could find few transformation combinations that could generalize to various models.

