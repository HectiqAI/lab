# Train a MNIST classifier

### Requirements
```
pyhectiqlab
torch
torchvision
tqdm
```

### Execution

If you don't already have an account, check the [setup documentaiton](https://docs.hectiq.ai/getting_started/quickstart/).

1. Open `main.py` and change the run project for your project.
```python
# Change the project on this line in main.py
run = Run(name="Train MNIST classifer - CNN network", project=project)
```

2. Execute the script
```
python main.py
```

### Inspect your model

Use a streamlit app to inspect your model. Copy the content of [app.py](./app.py) in an app.
[Click here](https://lab.hectiq.ai/public/hectiq-ai/demo/mnist-model) for a live demo.