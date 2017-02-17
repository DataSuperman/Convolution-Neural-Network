# Convolution-Neural-Network
CNN in Java

#Modifying the network
To modify the architecture just open the MyNeuralNetwork class under test folder. Just to clarify softmax doesn't seem to work 
properly.

# Training 
MyNeuralNetworkTrain class under the test folder, all the hyperparameters can be defined there. Just a thing this class saves 
all the weights under weights folder. Typically if a folder is named as "weights-16.20000" it means the network is stored for 
which validation accuracy was 16.20000% . Everytime one stops the network, weights are saved inside the folder and if you restart 
the  MyNeuralNetworkTrain class it will start from those weights. If you want to start from random initialization just delete
all 4 files under weights folder.

#Test
MyNeuralNetworkTest : This class does the forward pass only and  predicts the labels, output values are saved in in the "Yte.csv" 
file under test folder. One can specify which accuracy weight folder he/she wants to use simply by changing the name of the folder 
inside.


This project is best viewed with IntelliJ IDE.
