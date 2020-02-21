# CS520-Neural-Network-Colourizer

The purpose of this assignment is to demonstrate and explore some basic techniques in supervised learning and
computer vision. Given a gray scale image, the task is to add a splash of color to it. For this problem, we implemented and tested the following approaches.
1. Vanilla Neural Network - Direct mapping from a single gray-scale value gray to a corresponding color (r, g, b) on a pixel by pixel basis. 
2. Convolutional Neural Network - Mapping a set of gray values to a single (r, g, b) value, which is the color of the central pixel of the set.
3. Classification. Shifting from a regression problem to a discrete classification problem. Instead of trying to determine the exact color of the pixel, it maps to one colour out of a palette of K colors.

The approaches and the results are described in detail in the attached report. 
Information regaring the code files -

1. Neural_network.py: Vanilla Neural Network. 

2. Single_CNN_layer.ipynb: Convulational Nerual Network.

3. CNN_RGB.ipynb: Three separate networks for RGB value.

4. CNN_Stochastic_gradient_Descent.ipynb: Stochastic Gradient Descent of CNN

5. Classification.py: Classifying the pixels into K color palette and making the model classify the classes instead of the (r,g,b) values. 



