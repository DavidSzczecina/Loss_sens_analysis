import pytorch_model_summary as pms
import torch
import sys
from pathlib import Path

# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from enums.modelArchEnum import ModelType
from modules import conv_nn, multilayer_perceptron, resnet18, vision_transformer

#model: torch.nn.Module  = conv_nn.ConvNN()
#model = multilayer_perceptron.SimpleMLP()
#model = resnet18.create_ResNet18()
model = vision_transformer.CustomVisionTransformer()

#dummy tensor for mock training 
dummy_tensor: torch.Tensor = torch.zeros((1, 784))

#Get the model name 
model_type: ModelType = model.get_model_type()
print("Model used: ", model_type)
#Different dummy tensor based on the type of model being used
if(model_type == ModelType.MLP):
    dummy_tensor = torch.zeros((1, 784))
elif(model_type == ModelType.CNN or model_type == ModelType.RESNET18 or model_type == ModelType.VIT):
    dummy_tensor = torch.zeros((1, 1, 28, 28))


print(pms.summary(model, dummy_tensor, show_input = True))