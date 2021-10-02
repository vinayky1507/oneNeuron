# oneNeuron
oneNeuron | perceptron

##bash command used
git add . && git commit -m "readme  updated" && git push origin main

##add image
<img src="/plots/and.png" alt="AND plot" width="500" height="600">

##python code
```python 
def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    print(df)
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)
```

## datesets
x1|x2|y
-|-|-
0|0|0
0|0|1
1|0|0
1|1|1

##submodule
* point1
* point2
* point 3