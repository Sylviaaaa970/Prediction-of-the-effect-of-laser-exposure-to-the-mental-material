# Prediction-of-the-effect-of-laser-exposure-to-the-mental-material
*Team member: Xue Yang, Shunshun Hao*

## Introduction
In the study of the mental material, it is meaningful to make the smoothness within a very small scale, like the scale of nanometer. To improve the smoothness of the surface of some mental material, laser is a very common method. But the theory of physics and chemistry can not predict accurately the result of the laser exposure in such small scale by now. Thus, my teammate and I want to do a project about the prediction of the surface state of the material after exposure of the laser.    
Implement Environment: Pytorch 1.3,Python 3.6.8
##	Data Processing
One of my friends is doing study of this field and he has the true experiment data, and the data is the result of white light interferometry.   
The material in the experiment is a square flat mental board. In one group of data there are 2 files, one for before exposure and the other for after exposure. Every file is a matrix with 1024*1024 numbers and every number represents the height of the material surface in a square of 840nm*840nm. By now we have about 9 groups of data, and there will be more in future.  
We want to use the initial matrix of the mental material and 5 other parameters of the laser (laser power, laser frequency, pulse width of the laser, scanning speed, scanning line width) to calculate the state(height) of the material after exposure of the laser. And every matrix can be separate to lots of small squares, such as 20 dots*20 dots, 50 dots*50 dots etc. Also, if we use stride=10 dots(the distance between 2 small squares), than we could get about ten thousand samples from one group of data.
## Network Structure
One of reasons that the prediction is so hard is that the surface of the material is not totally smooth. For example, there are some cracks or bulges on the surface of the material, and as a result the height change of the small square around those things will be different with the smooth area.  
Thus, we want to use a CNN to find the features of the surface such as cracks or bulges, and learn their effect to squares around them. We pick a very common structure of CNN and use deconvolution layer to output a h_hat = same size as the original data:
Convolution layer1 – Active layer(Relu) – Convolution layer2 – Active layer(Relu) – add the effect of laser parameters – Deconvolution layer - Active layer(Relu)  

The loss function:   
Loss = mean_square(|| h_hat-h_true|| 2)  
Loss is the mean of || h_hat-h_true|| 2 of all squares in a batch  

The Matrics:  
Distance = mean_dot(|| h_hat-h_true|| 1)  
Distance is the mean of || h_hat-h_true|| 1 of all dots in test set 

h_true represent the true height after exposure, and h_hat represent the predicted height. 
 
## Best Result and Analysis
Loss = 0.3493  
Distance = 0.1355  
The prediction error is 0.1355 micrometer.

## Control Experiment
Experiment results:
|Order|Loss|Distance|Analysis|
|-|-|-|-|
|0|16.0084|\|
|1|0.3598|0.1386|
|2|10.3299|4.3138|Vanishing Gradient|
|3|10.3517|4.3230|Vanishing Gradient|
|4|0.6028|0.1371|
|5|9.8471|3.9344|Vanishing Gradient|
|6|34.5866|4.3320|Vanishing Gradient|
|7|0.3790|0.1457|
|8|0.3658|0.1419|
|9|0.3493|0.1355|Best Result|
|10|0.3649|0.1411|
|11|0.3699|0.1432|
|12|0.3769|0.1456|
|13|0.4815|0.1855|
|14|0.9173|0.3734|
|15|10.1501|4.1948|Vanishing Gradient|

### 1. Network Structure
|Order|Network Structure|Loss|Distance|
|-|-|-|-|
|7|one Convolution layer|0.3790|0.1457|
|1|two Convolution layer|0.3598|0.1386|
|8|three Convolution layer|0.3658|0.1419|
#### Other conditions:  
Square size:30  
Stride:16  
Normalization:Yes  
The way to use laser parameter: Each channel has one weight  
Number of channels of first convolution layer:16  
Batch size:1
#### Analysis:
The model with two Convolution layer is best. It is because the one with one Convolution layer is too simple to find correct features of the board surface and the one with three Convolution layer is overfiting and perform bad in test set. 

### 2. The way to use laser parameter
|Order|Network Structure|Loss|Distance|Analysis|
|-|-|-|-|-|
|9|Universal weight|0.3493|0.1355|Best Result|
|1|Each channel has one weight |0.3598|0.1386|
|5|Fully-connected layer for laser parameters|9.8471|3.9344|Vanishing Gradient|
|15|Convolution layer and Fully-connected layer for laser parameters|10.1501|4.1948||Vanishing Gradient|
#### Other conditions:  
Square size:30  
Stride:16  
Normalization:Yes  
The number of Convolution layer: Two  
Number of channels of first convolution layer:16  
Batch size:1
#### Analysis:
*Universal weight* means there is only one weight for the 5 laser parameters. Weight.dot(laser_parameter) is multiplied to the result of Convolution layer in every channel.   

*Each channel has one weight* means every channel has a weight for laser parameters. The reason to design this is that every channal is to learn a specific feature of the board surface and the laser parameters may have different effect in different channel. But the result is not very good, I guess it is about the lack of data because there are only 9 effect laser parameter samples.  

*Fully-connected layer for laser parameters* and  *Convolution layer and Fully-connected layer for laser parameters* are also the attempt to find the effect of laser parameters but the result are all bad. But I suppose with more data these complex models will perform better.

### 3. Normalization
|Order|Normalization|Loss|
|-|-|-|
|0|No|16.0084|
|1|Yes|0.3598|
#### Other conditions:  
Square size:30  
Stride:16  
The way to use laser parameter: Each channel has one weight   
The number of Convolution layer: Two  
Number of channels of first convolution layer:16  
Batch size:1
#### Analysis:
Obviously, when the data is normalized to range(0,1), the model perform much better.

### 4. Square size
|Order|Square size|Loss|Distance|Analysis|
|-|-|-|-|-|
|1|30|0.3598|0.1386|
|4|50|0.6028|0.1371|
|6|100|34.5866|4.3320|Vanishing Gradient|
#### Other conditions:   
Stride:16  
Normalization:Yes  
The way to use laser parameter: Each channel has one weight   
The number of Convolution layer: Two  
Number of channels of first convolution layer:16  
Batch size:1
#### Analysis:
The loss is the mean of every square and it is influenced by the square size, so we should compare distance here.
The model with 50 dots in one square is slightly better than the one with 30 dots. But the model with 100 dots has very bad performance. I suppose the reason is the drastic decrease of number of samples and these samples can not train a such complex model.

### 5. Stride
|Order|Stride|Loss|Distance|Analysis|
|-|-|-|-|-|
|10|10|0.3649|0.1411|
|9|16|0.3493|0.1355|Best Result|
|11|20|0.3699|0.1432|
#### Other conditions:   
Square size:30    
Normalization:Yes  
The way to use laser parameter: Universal weight  
The number of Convolution layer: Two
Number of channels of first convolution layer:16  
Batch size:1

### 6. Number of channels of first convolution layer
|Order|Number of channels|Loss|Distance|Analysis|
|-|-|-|-|-|
|13|8|0.4815|0.1855|
|9|16|0.3493|0.1355|Best Result|
|12|32|0.3769|0.1456|
#### Other conditions: 
Stride:16  
Square size:30    
Normalization:Yes  
The way to use laser parameter: Universal weight  
The number of Convolution layer: Two  
Batch size:1

### 7.Batch size
|Order|Batch size|Loss|Distance|Analysis|
|-|-|-|-|-|
|9|1|0.3493|0.1355|Best Result|
|14|100|0.9173|0.3734|
#### Other conditions: 
Stride:16  
Square size:30    
Normalization:Yes  
The way to use laser parameter: Universal weight  
The number of Convolution layer: Two  
Number of channels of first convolution layer:16


## Conclusion
The prediction error of the best model is about 0.1355 micrometers and the practical demand is 0.001 micrometers, so it still needs to be improved. There are several fields can be improved:  

(a) The way to use laser parameters. It is a trade-off of the complexity of the model. If it is too simple, it can't learn the correct effect to the board, but the lack of samples(I suppose the number of samples will be limited in one hundrud) limits the complexity. 

(b)The problem of Vanishing Gradient. Besides using more data, the methods such as regularization, batch normalization and PCA can also be tried.
