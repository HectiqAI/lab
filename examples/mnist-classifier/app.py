import os
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from pyhectiqlab.mlmodels import download_mlmodel
from pyhectiqlab import Config

mlmodel_name = "mnist-tiny-model"
version = "1.0.0"
project_path = "hectiq-ai/demo" # Change for your project

@st.cache
def download_contents():
    model_path = download_mlmodel(mlmodel_name=mlmodel_name, 
                 project_path=project_path, 
                 version=version, 
                 save_path='./')
    config = Config.load(os.path.join(model_path, "config.json"))
    return model_path, config

model_path, config = download_contents()

class TinyModel(nn.Module):

    def __init__(self, 
                 input_channels: Optional[int] = 1, 
                 out_channels: Optional[int] = 10,
                 dropout_prob: Optional[float] = 0.2,
                 depth: Optional[int] = 2, 
                 width: Optional[int] = 128):
        super(TinyModel, self).__init__()
        layers = [nn.Conv2d(input_channels, width, 3, 1)]
        for layer in range(depth-1):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(width, width, 3, 1))
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(width, out_channels))
        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)
    
    def predict(self, x):
        with torch.no_grad():
            y = self.sequence(x)
            return F.softmax(y, dim=-1)
      
model = TinyModel(**config.model)
model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
st.title('✍️ MNIST test')

with st.sidebar:
  st.header("Model")

  st.markdown(f"""
This is an app to inspect your model. The model has been trained and pushed with pyhectiqlab. 
  - Model: {mlmodel_name}
  - Version: {version}
  - Project: {project_path}
  """)
  
col1, col2 = st.columns(2)
with col1:
  SIZE = 300
  st.write('Draw something')
  canvas_result = st_canvas(
      fill_color='#000000',
      stroke_width=20,
      stroke_color='#FFFFFF',
      background_color='#000000',
      width=SIZE,
      height=SIZE,
      drawing_mode="freedraw",
      key='canvas')

with col2:
  if canvas_result.image_data is not None:
      img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
      rescaled = cv2.resize(img, (120, 120), interpolation=cv2.INTER_NEAREST)
      st.write('Model Input')
      st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = test_x.reshape(1, 1, 28, 28)
    x = np.asarray(x)
    x = torch.from_numpy(x).float()/255
    transform = torchvision.transforms.Normalize((config.normalize.mean, ), (config.normalize.std, ))
    x = transform(x)
    scores = model.predict(x)
    label = int(scores.argmax(-1))
    score = scores[0][label]
    st.header(f'Label: {label} ({score:.2%})')