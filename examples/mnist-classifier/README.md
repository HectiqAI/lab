# Train a MNIST classifier
[Live app for inspection](https://lab.hectiq.ai/public/hectiq-ai/demo/mnist-model)

### Requirements
```
pyhectiqlab
torch
torchvision
tqdm
```

### Execution

If you don't already have an account, check the [setup documentaiton](https://docs.hectiq.ai/getting_started/quickstart/).

1. Open `main.py` and set `project`.
2. Execute the script
```
python main.py
```

### Inspect your model

Use a streamlit app to inspect your model. Create an app in your project (web application) and add the files.
1. Copy the content of [app.py](./app.py) in the app files.
2. Copy the requirements below in your `requirements.txt` of the app:
```
streamlit
numpy
matplotlib
torch
torchvision
opencv-python-headless
streamlit-drawable-canvas
```

A [public demo](https://lab.hectiq.ai/public/hectiq-ai/demo/mnist-model) is available.