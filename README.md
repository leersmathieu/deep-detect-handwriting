
# Deep Detect Handwriting

Deep detect my handwriting is a app based on a CNN model, to recognize the digit that you draw on the canvas.  
You can check the model construction [here](https://github.com/leersmathieu/deep-detect-handwriting/blob/master/notebooks/CNN_mnist_model.ipynb)

# Try it yourself !

On my serveur http://deepwriting.tamikara.xyz/

OR

With *docker*


For try this app on your computer with docker, just enter this command in your terminal.   
It's magic.

```docker
docker run -p 5000:5000 leersma/deep-detect-handwriting:latest
```
just pay attention to the port used
If you run it on port 5000 as in the example, you can easily access it like this: http://localhost:5000/.
