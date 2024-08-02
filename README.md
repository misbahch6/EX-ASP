# EX-ASP Explainable AI using Answer Set Programming

**Scripts** to convert Multi-Layer Perceptrons (MLPs) into Answer Set Programming (ASP) to facilitate the computation of explanations. It involves representing the MLP in the form of a logic program. The goal is to compute subset minimal explanations for the model's predictions. 

# Steps to run the script
- Specify weights, biases, number of layers and number of hidden neurons.
- Input data and predictions

```
python converter.py [folder]  
```
Replace `[folder]` with the path to the directory containing the data and model files.  
This script will write both instance and encodings part of the ASP program. 
Then, it use deletion based algorithm to calculate explanation for the given instance.  
  
To calculate the explanation for another instance, please customise the source file.

# Requirements
clingo  
clingolpx  
numpy  
pandas
