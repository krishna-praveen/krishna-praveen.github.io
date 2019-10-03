# Keras basic implementation
## Advantages of using keras

- User friendly
- Modular and composable
- Easy to extend


[tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) is the implementation of the keras API in the tensorflow.

This also supports the [eager execution](https://www.tensorflow.org/guide/keras#eager_execution) as well as [tf.data ](https://www.tensorflow.org/api_docs/python/tf/data)modules for data pipeline and processing and then finally [estimators](https://www.tensorflow.org/guide/estimators) which are used to maintatin structure to the training code for machine learning tasks.


```
!pip install -q pyyaml  # Required to save models in YAML format

```


```
# To maintaing the code compatibility with other python versions
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

# The tf.keras version might not be the same as latest keras version from pyPI
print(tf.version.VERSION)
print(tf.keras.__version__)
```

    1.14.0
    2.2.4-tf


The keras format when saving models is defaulted to checkpoint format. We have to specify what type of checkpoint format we want to use , for example if we want to use hdf5 we have to specify it as ```save_format=h5``` in the code.

## Building a simple keras model

### Sequential model


```
# Creating a sample model here
model = tf.keras.Sequential()

# Adding layers to the above model that is made
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64,activation="relu"))

# Finally adding the final layers with 10 units of FC units
model.add(layers.Dense(10,activation="softmax"))

```

**Configure the layers**

There are many tf.keras.layers available with some common constructor parameters:

- *activation*: Set the activation function for the layer. This parameter is specified by the name of a built-in function or as a callable object. By default, no activation is applied.

- *kernel_initializer* and bias_initializer: The initialization schemes that create the layer's weights (kernel and bias). This parameter is a name or a callable object. This defaults to the "Glorot uniform" initializer.

- *kernel_regularizer* and bias_regularizer: The regularization schemes that apply the layer's weights (kernel and bias), such as L1 or L2 regularization. By default, no regularization is applied.


```
## Different examples of how the configuring of the layers works
layers.Dense(64, activation="sigmoid")
# OR the below one works the same
layers.Dense(64,activation=tf.sigmoid)

# Using kernel_initializer and bias_initializer
# This will make that all the layer is intialized with a random orthogonall
#   matrix
layers.Dense(64, kernel_initializer="orthogonal")

# using the bias_initalizer in the same way
# Here the bias initalizer is invoked and all the bias are set to 2.0 constant
layers.Dense(64,bias_initializer=tf.keras.initializers.constant(2.0))

## Using kernel_regularizer and bias_regularizer
# We are using the kernel_regularizer and setting the linaer layer
#       with a L1 regularization of 0.01 and it is applied to kernel_matrix
layers.Dense(64,kernel_regularizer=tf.keras.regularizers.l1(0.01))

# Same as above but we are using L2 regularization here
layers.Dense(64,bias_regularizer=tf.keras.regularizers.l2(0.01))
```




    <tensorflow.python.keras.layers.core.Dense at 0x7fbf2ab98ba8>



## Training and Evaluating



### Setting up training
After the model is configured we can configure its learning process by using the compile method on it


```
# We can give all the layers in a format of list here inside 
#   instead of adding it with model.add
model = tf.keras.Sequential([
        # Adding a layer with 64 units of dense layer
        # WE have to declare input_shape here.
        layers.Dense(64, activation="relu", input_shape=(32,)),
        # Another layer with same dense units
        layers.Dense(64, activation="relu"),

        layers.Dense(10, activation='softmax')
     ])

## Compiling
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```

tf.keras.Model.compile takes three important arguments:

- *optimizer*: This object specifies the training procedure. Pass it optimizer instances from the tf.train module, such as tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, or tf.train.GradientDescentOptimizer.

- *loss*: The function to minimize during optimization. Common choices include mean square error (mse), categorical_crossentropy, and binary_crossentropy. Loss functions are specified by name or by passing a callable object from the tf.keras.losses module.

- *metrics*: Used to monitor training. These are string names or callables from the tf.keras.metrics module.


```
# Sample examples of having different optimizers and loss functions
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              # Mean squared error
              loss='mse',
              # Mean absolute error metrics
              metrics = ['mae'])

# Instead of specifying the names for the losses and metrics
#   We can use the tf.keras module to tab suggest the required modules.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```

### Input numpy data

For small memory datasets we can use the in memory numpy arrays and use the fit method to fit the data in the model


```
import numpy as np

def random_one_hot_labels(shape):
    n, n_class = shape
    # Getting all the indexes of the classes, or randomly assiging them
    classes = np.random.randint(0,n_class,n)
    labels = np.zeros((n,n_class))

    # For all the array and for each row we are assining the randomly selected
    #   class label from above to set it equal to 1, rest all will be zero
    labels[np.arange(n),classes]=1
    return labels

data = np.random.random((1000,32))
labels = random_one_hot_labels((1000,10))

model.fit(data, labels, epochs=10,batch_size=32)
```

    Epoch 1/10
    1000/1000 [==============================] - 0s 252us/sample - loss: 2.3234 - categorical_accuracy: 0.0930
    Epoch 2/10
    1000/1000 [==============================] - 0s 134us/sample - loss: 2.3001 - categorical_accuracy: 0.1020
    Epoch 3/10
    1000/1000 [==============================] - 0s 120us/sample - loss: 2.2961 - categorical_accuracy: 0.1070
    Epoch 4/10
    1000/1000 [==============================] - 0s 119us/sample - loss: 2.2886 - categorical_accuracy: 0.1350
    Epoch 5/10
    1000/1000 [==============================] - 0s 121us/sample - loss: 2.2843 - categorical_accuracy: 0.1460
    Epoch 6/10
    1000/1000 [==============================] - 0s 124us/sample - loss: 2.2731 - categorical_accuracy: 0.1340
    Epoch 7/10
    1000/1000 [==============================] - 0s 123us/sample - loss: 2.2758 - categorical_accuracy: 0.1370
    Epoch 8/10
    1000/1000 [==============================] - 0s 127us/sample - loss: 2.2545 - categorical_accuracy: 0.1490
    Epoch 9/10
    1000/1000 [==============================] - 0s 143us/sample - loss: 2.2590 - categorical_accuracy: 0.1580
    Epoch 10/10
    1000/1000 [==============================] - 0s 140us/sample - loss: 2.2320 - categorical_accuracy: 0.1760





    <tensorflow.python.keras.callbacks.History at 0x7fbf2a8f1b00>



tf.keras.Model.fit takes three important arguments:

- *epochs*: Training is structured into epochs. An epoch is one iteration over the entire input data (this is done in smaller batches).

- *batch_size*: When passed NumPy data, the model slices the data into smaller batches and iterates over these batches during training. This integer specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the batch size.

- *validation_data*: When prototyping a model, you want to easily monitor its performance on some validation data. Passing this argument—a tuple of inputs and labels—allows the model to display the loss and metrics in inference mode for the passed data, at the end of each epoch.


```
# Example using the validation data
import numpy as np
data = np.random.random((1000,32))
labels = random_one_hot_labels((1000,10))

val_data = np.random.random((100,32))
val_labels = random_one_hot_labels((100,10))

model.fit(data,labels,epochs=10,validation_data=(val_data,val_labels))
```

    Train on 1000 samples, validate on 100 samples
    Epoch 1/10
    1000/1000 [==============================] - 0s 410us/sample - loss: 2.3351 - categorical_accuracy: 0.0900 - val_loss: 2.3016 - val_categorical_accuracy: 0.1500
    Epoch 2/10
    1000/1000 [==============================] - 0s 140us/sample - loss: 2.2990 - categorical_accuracy: 0.1180 - val_loss: 2.3053 - val_categorical_accuracy: 0.0800
    Epoch 3/10
    1000/1000 [==============================] - 0s 135us/sample - loss: 2.2971 - categorical_accuracy: 0.1220 - val_loss: 2.2904 - val_categorical_accuracy: 0.1100
    Epoch 4/10
    1000/1000 [==============================] - 0s 138us/sample - loss: 2.2889 - categorical_accuracy: 0.1450 - val_loss: 2.3342 - val_categorical_accuracy: 0.0600
    Epoch 5/10
    1000/1000 [==============================] - 0s 149us/sample - loss: 2.2806 - categorical_accuracy: 0.1500 - val_loss: 2.3397 - val_categorical_accuracy: 0.0400
    Epoch 6/10
    1000/1000 [==============================] - 0s 153us/sample - loss: 2.2704 - categorical_accuracy: 0.1470 - val_loss: 2.3562 - val_categorical_accuracy: 0.0600
    Epoch 7/10
    1000/1000 [==============================] - 0s 149us/sample - loss: 2.2740 - categorical_accuracy: 0.1410 - val_loss: 2.3809 - val_categorical_accuracy: 0.0600
    Epoch 8/10
    1000/1000 [==============================] - 0s 148us/sample - loss: 2.2635 - categorical_accuracy: 0.1360 - val_loss: 2.3285 - val_categorical_accuracy: 0.0600
    Epoch 9/10
    1000/1000 [==============================] - 0s 144us/sample - loss: 2.2397 - categorical_accuracy: 0.1640 - val_loss: 2.3796 - val_categorical_accuracy: 0.0700
    Epoch 10/10
    1000/1000 [==============================] - 0s 162us/sample - loss: 2.2403 - categorical_accuracy: 0.1720 - val_loss: 2.4035 - val_categorical_accuracy: 0.1000





    <tensorflow.python.keras.callbacks.History at 0x7fbf2b9a25f8>



### Input tf.data datasets

We can use the [Datasets.API](https://www.tensorflow.org/guide/datasets) to scale the training the large scale or use even multi- device training. We can use the tf.data.Dataset instance to the fit method.


```
# Using the above data and labels we can use that to showcase what
#   our tf.data can do
dataset = tf.data.Dataset.from_tensor_slices((data,labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# THe dataset has been converted to tf.data format and also it has been
#   dvidied in to batch size of 32 and the full data is also repeated

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
# This means the number of times the trainig is done on the current epoch before
#      moving to the next epoch. Since the dataset has batch size inbuilt in it
#       we dont need to specify it again in the model.fit
model.fit(dataset, epochs = 10, steps_per_epoch=30)
```

    WARNING:tensorflow:Expected a shuffled dataset but input dataset `x` is not shuffled. Please invoke `shuffle()` on input dataset.
    Epoch 1/10
    30/30 [==============================] - 0s 16ms/step - loss: 2.2374 - categorical_accuracy: 0.1719
    Epoch 2/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.2184 - categorical_accuracy: 0.1720
    Epoch 3/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.2254 - categorical_accuracy: 0.1688
    Epoch 4/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1967 - categorical_accuracy: 0.1806
    Epoch 5/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1883 - categorical_accuracy: 0.1816
    Epoch 6/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1686 - categorical_accuracy: 0.1923
    Epoch 7/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1685 - categorical_accuracy: 0.1891
    Epoch 8/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1752 - categorical_accuracy: 0.1912
    Epoch 9/10
    30/30 [==============================] - 0s 5ms/step - loss: 2.1932 - categorical_accuracy: 0.1741
    Epoch 10/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1712 - categorical_accuracy: 0.1795





    <tensorflow.python.keras.callbacks.History at 0x7fbf2a738c18>



Datasets can also be used for validation


```
dataset = tf.data.Dataset.from_tensor_slices((data,labels))
# We can chain the functions to resembel and perform the same actions as above
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_labels))
val_dataset = val_dataset.batch(32).repeat()

# Here we can specify the val_dataset which is same format as dataset
# We no longer need to specify labels again since they are already included in the
#   dataset format when we are loading them. But one thing to rememeber is that
#   the data,labels are to be given in tuple format and the order is important.
model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data= val_dataset,
          validation_steps=3)
```

    WARNING:tensorflow:Expected a shuffled dataset but input dataset `x` is not shuffled. Please invoke `shuffle()` on input dataset.
    Epoch 1/10
    30/30 [==============================] - 1s 26ms/step - loss: 2.2115 - categorical_accuracy: 0.1625 - val_loss: 2.3502 - val_categorical_accuracy: 0.1458
    Epoch 2/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.2076 - categorical_accuracy: 0.1613 - val_loss: 2.3425 - val_categorical_accuracy: 0.0833
    Epoch 3/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.2275 - categorical_accuracy: 0.1432 - val_loss: 2.3868 - val_categorical_accuracy: 0.1146
    Epoch 4/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.2097 - categorical_accuracy: 0.1592 - val_loss: 2.4244 - val_categorical_accuracy: 0.0833
    Epoch 5/10
    30/30 [==============================] - 0s 5ms/step - loss: 2.1771 - categorical_accuracy: 0.1656 - val_loss: 2.4497 - val_categorical_accuracy: 0.0938
    Epoch 6/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1757 - categorical_accuracy: 0.1720 - val_loss: 2.4626 - val_categorical_accuracy: 0.0833
    Epoch 7/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1725 - categorical_accuracy: 0.1934 - val_loss: 2.4493 - val_categorical_accuracy: 0.0938
    Epoch 8/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1672 - categorical_accuracy: 0.1934 - val_loss: 2.4550 - val_categorical_accuracy: 0.1146
    Epoch 9/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1280 - categorical_accuracy: 0.2051 - val_loss: 2.4939 - val_categorical_accuracy: 0.1042
    Epoch 10/10
    30/30 [==============================] - 0s 4ms/step - loss: 2.1067 - categorical_accuracy: 0.2073 - val_loss: 2.4524 - val_categorical_accuracy: 0.1042





    <tensorflow.python.keras.callbacks.History at 0x7fbf2a6c79b0>



### Evaluate and predict

The [tf.keras.Model.evaluate](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) and [tf.keras.Model.predict](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict) methods can use NumPy data and a tf.data.Dataset.

To evaluate the inference-mode loss and metrics for the data provided:


```
data = np.random.random((1000,32))
labels = random_one_hot_labels((1000,10))

# Evaluate will evaluate the data with the given labels and will give us the
#   loss at this current model step and also we can give batch_size to make it
#   evaluate at faster level
model.evaluate(data,labels, batch_size=32)
```

    1000/1000 [==============================] - 0s 201us/sample - loss: 2.4434 - categorical_accuracy: 0.1030





    [2.4434038658142088, 0.103]




```
## Now to predict
result = model.predict(data,batch_size=32)
print(result.shape)
```

    (1000, 10)


## Building Advanced Models

### Functional API

Using this [keras funcitonal](https://keras.io/getting-started/functional-api-guide/) api we can build more complex models.
Generally we are seeing Sequential Model until now . But we can make multiinput and multi ouput models as well.

We can make the following type of models
- Multi-input models,
- Multi-output models,
- Models with shared layers (the same layer called several times),
- Models with non-sequential data flows (e.g. residual connections)


Building a model with the functional API works like this:

- A layer instance is callable and returns a tensor.
- Input tensors and output tensors are used to define a tf.keras.Model instance.
- This model is trained just like the Sequential model.

The following example uses the functional API to build a simple, fully-connected network:


```
# This is a place holder tensor for the input we are going to pass
# Generally for a image the input shape will be of this format
# Also remember that we pass the image as batches and we have extra 
#   shape value of the batch size that awe are going. 
#   Finally the shape that we are going to have is (None, height,width,channels)
#   
inputs = tf.keras.Input(shape=(32,))

# A keras layer is callable on this inputs , so we make a layer and pass this
#   inputs to the keras layer
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation = "relu")(x)
# Using the Keras API we can chain the layers one after another and pass the inputs
#   to each layer
predictions = layers.Dense(10, activation="softmax")(x)

```


```
# Now we can instantiate the model by passing the arguments to the model
#   as inputs and outputs
sample_model = tf.keras.Model(inputs = inputs, outputs = predictions)

# Now we can apply the above functions we have used to compile , give
#   optimizer and loss functions and also we can use what metrics to be stated
# We can use any kind of optimizer we want and also we can use which kind of
#   loss to be stated
sample_model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Final step of fitting to be implemented now as the model is ready now
sample_model.fit(x=data, 
                y=labels, 
                batch_size=32, 
                epochs =5)

# If we use the tf.data module we dont need to specify the batch_size if we
#   have already created a pipeline out of it.
# Like we did above in the Input #####tf.data datasets#####
```

    Epoch 1/5
    1000/1000 [==============================] - 0s 284us/sample - loss: 2.3320 - acc: 0.1040
    Epoch 2/5
    1000/1000 [==============================] - 0s 127us/sample - loss: 2.2957 - acc: 0.1350
    Epoch 3/5
    1000/1000 [==============================] - 0s 141us/sample - loss: 2.2908 - acc: 0.1410
    Epoch 4/5
    1000/1000 [==============================] - 0s 144us/sample - loss: 2.2890 - acc: 0.1210
    Epoch 5/5
    1000/1000 [==============================] - 0s 137us/sample - loss: 2.2804 - acc: 0.1320





    <tensorflow.python.keras.callbacks.History at 0x7fbf2a57a908>



### Model-Subclassing

Sort of like making class and then use the Model object inside it to create and add any function or make any new model out of it.

We can build a fully customizable model by subclassing the tf.keras.Model class and then we can define our own forward pass.

We can create layers in the __init__ method and set them as attributes of the class instance.

Also we have to define the pass in the ```call``` method.

Model subclassing is useful when the ```eager_execution``` is enabled since forward pass can be used imperatively


```
## Using subclassing and using the tf.keras.Model inside it

class SampleModel(tf.keras.Model):

    def __init__(self,numclasses = 10):
        super(SampleModel,self).__init__(name="samplemodel")
        self.numclasses = numclasses
        # Creating the first layer and second layer here
        # We are not chaining layers or adding them in series
        #     WE are just declaring them
        self.firstlayer = layers.Dense(64, activation="relu")
        self.secondlayer = layers.Dense(numclasses, activation="softmax")
    
    def call(self, inputs):
        # WE are going to define our forward pass here
        # THis is how we are going to do it. First in init we initialize
        #   the required layers and parameters that we have to pas
        # THen we are going to use those layers and call them and create
        #   the graph or we can say newural netwrok required
        x = self.firstlayer(inputs)
        x = self.secondlayer(x)
        return x
    
    def compute_output_shape(self, input_shape):
        # Honestly i dont know what this method does. It says that
        # If we want to use the subclassed model we have to override this function
        #   as part of functional style model. Or else this method is optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        
        # WE are setting the last shape of the output so that it matches our 
        #   output shape which is number of classes we want as outputs
        # Here the last value of the shape that nee tdo be modified is changed to
        #   the shape that we require
        return tf.TensorShape(shape)
```

Instantiating the model that we have created from the sample model which is made using the [subclassing](#Model-Subclassing) method.


```
modelhere = SampleModel(10)

# Now we need to compile the model after instantiaing it.
# Normal compiling of model by stating the optimizer we want to use
#   along with losses that we need to consdier and also the metrics
#   we want to measure
modelhere.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics = [tf.keras.metrics.categorical_accuracy])

# After compiling and adding everything that is required the final step
#   is to fit the model with the data we want to use. We may or maynot
#   need to specify batchsize depending on the data we are using. If we are
#   using the tf.data.datasets we dont need to use the batch_size if we have
#   specified it.
modelhere.fit(x=data, y = labels, batch_size=32, epochs=5)
```

    Epoch 1/5
    1000/1000 [==============================] - 0s 269us/sample - loss: 2.3471 - categorical_accuracy: 0.1160
    Epoch 2/5
    1000/1000 [==============================] - 0s 113us/sample - loss: 2.3036 - categorical_accuracy: 0.1190
    Epoch 3/5
    1000/1000 [==============================] - 0s 121us/sample - loss: 2.2956 - categorical_accuracy: 0.1400
    Epoch 4/5
    1000/1000 [==============================] - 0s 130us/sample - loss: 2.2898 - categorical_accuracy: 0.1370
    Epoch 5/5
    1000/1000 [==============================] - 0s 117us/sample - loss: 2.2810 - categorical_accuracy: 0.1490





    <tensorflow.python.keras.callbacks.History at 0x7fbf2b83dd30>



### Creating CUSTOM layers
We can create custom layers after using and subclassing the [tf.keras.layers.Layer]() and then implementing the following methods on it.

- **build** : Create the weights of the layer. Add the weights to that layer with add_weights method
- **call** : Propagate the forward pass using the layers we have created in the init and build method
- **compute_output_shape** : Specify how to compute the output_shape given the input shape
- Optionally a layer can be serialized by implementing the get_config method from the from_config class

Following below is an example of implementing a custom layer that implements a matmul of an input with a kernel matrix


```
class SampleLayerModel(tf.keras.layers.Layer):

    # Defining the initaliziation with the num of output classes
    # Remember that we can add more arguments not only this one thing
    
    # Infact we can use kwargs ( aka key worded arguments) and then
    #   define a variable inside that stores each of these kwarg value
    def __init__(self,num_classes,**kwargs):
        self.num_classes = num_classes
        super(SampleLayerModel,self).__init__(**kwargs)

    def build(self,input_shape):

        # Dont forget the double brackets here, since the shape is a tuple
        shape = tf.TensorShape((input_shape[1], self.num_classes))

        # Now creating a trainable weights for this layer
        self.kernel = self.add_weight(name="kernelhere",
                                      shape=shape,
                                      initializer = "uniform",
                                      trainable=True)
        
        # After adding a kernel and doing this, we have to call the
        #   build method at the end of this method
        super(SampleLayerModel,self).build(input_shape)
    
    # After building the model this is what that is returned finally as the 
    #   output
    def call(self,inputs):
        return tf.matmul(inputs, self.kernel)

    
    # Finally computing the output_shape
    # The input_shape is the image or data or whatever the previous layer shape
    #   that is being passed on to this layer.

    def compute_output_shape(self,input_shape):
        # Taking the shape of the input and converting it to list
        shape = tf.TensorShape(input_shape).as_list()

        # Now taking the last value of the inputshape which indicates the channels
        # And making it equal to the output_classes that we are actually going to output
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(SampleLayerModel, self).get_config()
        base_config['output_dim'] = self.num_classes
        return base_config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


```


```
# We are creating a sequential graph and in this we are adding our own
#   custom layer that we have created. In that we have 10 classes that is value
#   passed as the output number of nodes that we are intereseted in.
modelhere = tf.keras.Sequential([
                                 SampleLayerModel(10),
                                 layers.Activation('softmax')
                                 ])

# Compiling the model or else we cannot call summary on this
# We can use two different kind of optimizers like we have said eariler
# Either :
#       - tf.keras.optimizers. whatever you want
#    or - tf.train. whatever you want

modelhere.compile(optimizer= tf.train.RMSPropOptimizer(0.001),
                  loss= tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

# Training for 5 epochs using the custom layer we have built
modelhere.fit(data, labels, batch_size=32, epochs=5)

```

    Epoch 1/5
    1000/1000 [==============================] - 0s 262us/sample - loss: 2.3045 - categorical_accuracy: 0.0940
    Epoch 2/5
    1000/1000 [==============================] - 0s 104us/sample - loss: 2.3032 - categorical_accuracy: 0.1060
    Epoch 3/5
    1000/1000 [==============================] - 0s 101us/sample - loss: 2.3008 - categorical_accuracy: 0.1200
    Epoch 4/5
    1000/1000 [==============================] - 0s 106us/sample - loss: 2.2973 - categorical_accuracy: 0.1250
    Epoch 5/5
    1000/1000 [==============================] - 0s 99us/sample - loss: 2.2948 - categorical_accuracy: 0.1390





    <tensorflow.python.keras.callbacks.History at 0x7fbf288bc4e0>



 ## CallBacks
 These are the custom objects that are passed to the keras model while the model is training. These objects can modify the behaviour of the model while it is training or after it. We can also write our own custom callback using the keras API.

Or 

We can use the [tf.keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) that include these:

- [tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) : This one saves the checkpoints of the model in regular intervals
- [tf.keras.callbacks.LearningRateScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler) : This will let us modify the learning rate (which is a hyperparameter) during the training where we can dynamically change the learning rate
- [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) : We can use this to stop the training of the model based on the conditions we can define. Here what we do is we define a criteria like some accuracy or some metric and monitor it and if it satisfies a condition then we ask the model to stop the training.
- [tf.keras.callbacks.TensorBoard](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) : Lets us monitor the model behaviour using the TensorBoard.

In order to use the tf.keras.callback we have to pass the callback in the fit method that we finally execute.


```
# Definng the callbacks list here with multiple callbacks
# Let us define Early stopping and make the model to stop training when
#   the metric score doesnt changes in 2 epochs
callbacks = [
             # Stops the model training when the validation loss doesn't improve
             #  in 2 epochs
             tf.keras.callbacks.EarlyStopping(monitor='val_loss' , patience=2),

             # Also log the entire thing to TensorBoard './logs' directory
             tf.keras.callbacks.TensorBoard(log_dir='./logs')
             ]

modelhere.fit(data, labels, batch_size=32, epochs=5, callbacks = callbacks, 
              validation_data = (val_data, val_labels))           
```

    Train on 1000 samples, validate on 100 samples
    Epoch 1/5
    1000/1000 [==============================] - 0s 433us/sample - loss: 2.2932 - categorical_accuracy: 0.1350 - val_loss: 2.3206 - val_categorical_accuracy: 0.1000
    Epoch 2/5
    1000/1000 [==============================] - 0s 119us/sample - loss: 2.2901 - categorical_accuracy: 0.1570 - val_loss: 2.3154 - val_categorical_accuracy: 0.1000
    Epoch 3/5
    1000/1000 [==============================] - 0s 120us/sample - loss: 2.2880 - categorical_accuracy: 0.1490 - val_loss: 2.3138 - val_categorical_accuracy: 0.1200
    Epoch 4/5
    1000/1000 [==============================] - 0s 129us/sample - loss: 2.2861 - categorical_accuracy: 0.1320 - val_loss: 2.3166 - val_categorical_accuracy: 0.1100
    Epoch 5/5
    1000/1000 [==============================] - 0s 128us/sample - loss: 2.2844 - categorical_accuracy: 0.1450 - val_loss: 2.3154 - val_categorical_accuracy: 0.0900





    <tensorflow.python.keras.callbacks.History at 0x7fbf2c20c780>



### Using tensorboard in Google Colab
Using the package tensorboardcolab we can enable the colab to visualize that sweet juicy graphs. To do that we have to import two modules from this package.
And pass the callback in the fit function to enable this.

Also after executing the cell code below we can see that a link is generated which is the tensorboard link. We can see our model details and etc in that.


```
!pip install tensorboardcolab

from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
tbc = TensorBoardColab()
```

    Requirement already satisfied: tensorboardcolab in /usr/local/lib/python3.6/dist-packages (0.0.22)
    Wait for 8 seconds...
    TensorBoard link:
    https://126578d0.ngrok.io



```
# Running the model again but here with enabling the tensorboard colab callback
callbacks = [
             # Stops the model training when the validation loss doesn't improve
             #  in 2 epochs
             tf.keras.callbacks.EarlyStopping(monitor='val_loss' , patience=2),

             # Also log the entire thing to TensorBoard './logs' directory
             TensorBoardColabCallback(tbc)
             ]

modelhere.fit(data, labels, batch_size=32, epochs=5, callbacks = callbacks, 
              validation_data = (val_data, val_labels))
      
```

    Train on 1000 samples, validate on 100 samples
    Epoch 1/5
    1000/1000 [==============================] - 0s 130us/sample - loss: 2.2826 - categorical_accuracy: 0.1430 - val_loss: 2.3206 - val_categorical_accuracy: 0.0800
    Epoch 2/5
    1000/1000 [==============================] - 0s 118us/sample - loss: 2.2793 - categorical_accuracy: 0.1620 - val_loss: 2.3148 - val_categorical_accuracy: 0.1100
    Epoch 3/5
    1000/1000 [==============================] - 0s 117us/sample - loss: 2.2780 - categorical_accuracy: 0.1560 - val_loss: 2.3173 - val_categorical_accuracy: 0.0900
    Epoch 4/5
    1000/1000 [==============================] - 0s 118us/sample - loss: 2.2758 - categorical_accuracy: 0.1690 - val_loss: 2.3138 - val_categorical_accuracy: 0.0800
    Epoch 5/5
    1000/1000 [==============================] - 0s 115us/sample - loss: 2.2743 - categorical_accuracy: 0.1650 - val_loss: 2.3177 - val_categorical_accuracy: 0.0800





    <tensorflow.python.keras.callbacks.History at 0x7fbf2878fdd8>



## Saving and Restoring weights
IMPORTANT


### Weights Only :
To save the weights we can use the [tf.keras.Model.save_weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights) :


```
modelhere = tf.keras.Sequential([
                                 layers.Dense(64, input_shape=(32,), activation="relu"),
                                 layers.Dense(10, activation='softmax')
                                ])

modelhere.compile(optimizer= tf.train.RMSPropOptimizer(0.001),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
```

Saving weights to a tensorflow checkpoint model


```
modelhere.save_weights('./weights/model_here')
```

Restoring the model weights is as easy as storing them , we just have to call load_weights
But remember that the model should match or else it wont work


```
modelhere.load_weights('./weights/model_here')
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fbf2793a518>



By default the weights are stored in a checkpoint format of tensorflow. But we can overwrite this scenario and specify other types of format to save the weights such as [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format. This format is the default format of keras model but here in tensorflow this is what we use


```
## Saving weights in HDF5
modelhere.save_weights('./weights/modelhere_newformat.h5', save_format='h5')

## And loading weights in HDF5
modelhere.load_weights('./weights/modelhere_newformat.h5')

```

### Configuration Only
Earlier we have seen how to save weights, we can also save the configuration of the model as well.
The model can be serialized without weights. And that saved configuration can recreate the model and the weights can be calculated or loaded as per requirement. Even without the code that defines the model we can recreate it if the configuration is loaded correctly

Keras supports [JSON](https://www.w3schools.com/js/js_json_intro.asp) and [YAML](https://en.wikipedia.org/wiki/YAML) [serialization](https://en.wikipedia.org/wiki/Serialization) formats:


#### Saving model and loading it using JSON


```
## We can serialize the model here using JSON format:
json_model_string = modelhere.to_json()
json_model_string
```




    '{"class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "Dense", "config": {"name": "dense_74", "trainable": true, "batch_input_shape": [null, 32], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_75", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'




```
## We can see that json string of the model can be checked here
# To print it in good format which is readable by humans (unless you are a bot)
import json
import pprint
pprint.pprint(json.loads(json_model_string))
```

    {'backend': 'tensorflow',
     'class_name': 'Sequential',
     'config': {'layers': [{'class_name': 'Dense',
                            'config': {'activation': 'relu',
                                       'activity_regularizer': None,
                                       'batch_input_shape': [None, 32],
                                       'bias_constraint': None,
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {'dtype': 'float32'}},
                                       'bias_regularizer': None,
                                       'dtype': 'float32',
                                       'kernel_constraint': None,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'dtype': 'float32',
                                                                         'seed': None}},
                                       'kernel_regularizer': None,
                                       'name': 'dense_74',
                                       'trainable': True,
                                       'units': 64,
                                       'use_bias': True}},
                           {'class_name': 'Dense',
                            'config': {'activation': 'softmax',
                                       'activity_regularizer': None,
                                       'bias_constraint': None,
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {'dtype': 'float32'}},
                                       'bias_regularizer': None,
                                       'dtype': 'float32',
                                       'kernel_constraint': None,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'dtype': 'float32',
                                                                         'seed': None}},
                                       'kernel_regularizer': None,
                                       'name': 'dense_75',
                                       'trainable': True,
                                       'units': 10,
                                       'use_bias': True}}],
                'name': 'sequential_21'},
     'keras_version': '2.2.4-tf'}



```
# Creating a new model which can be built by using the json string that we have
#   created. This will load the model and creates it but it wont have any weights
# Although the weights can be loaded from the load_weights function
newmodel = tf.keras.models.model_from_json(json_model_string)
```

 #### Saving model and loading it using YAML   


```
yaml_model_string = modelhere.to_yaml()

# We dont need any extra pprint functions to print this in
#   new format.
print(yaml_model_string)
```

    backend: tensorflow
    class_name: Sequential
    config:
      layers:
      - class_name: Dense
        config:
          activation: relu
          activity_regularizer: null
          batch_input_shape: !!python/tuple [null, 32]
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {dtype: float32}
          bias_regularizer: null
          dtype: float32
          kernel_constraint: null
          kernel_initializer:
            class_name: GlorotUniform
            config: {dtype: float32, seed: null}
          kernel_regularizer: null
          name: dense_74
          trainable: true
          units: 64
          use_bias: true
      - class_name: Dense
        config:
          activation: softmax
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {dtype: float32}
          bias_regularizer: null
          dtype: float32
          kernel_constraint: null
          kernel_initializer:
            class_name: GlorotUniform
            config: {dtype: float32, seed: null}
          kernel_regularizer: null
          name: dense_75
          trainable: true
          units: 10
          use_bias: true
      name: sequential_21
    keras_version: 2.2.4-tf
    



```
### Caution: Subclassed models are not serializable because their architecture 
#   is defined by the Python code in the body of the call method.

newmodelyaml = tf.keras.models.model_from_yaml(yaml_model_string)
newmodelyaml.summary()
```

    Model: "sequential_21"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_74 (Dense)             (None, 64)                2112      
    _________________________________________________________________
    dense_75 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 2,762
    Trainable params: 2,762
    Non-trainable params: 0
    _________________________________________________________________


### Entire Model
We have seen the model is saved using the Json and other formats and also seen the weights are also saved here. We can also save the entire thing including weights, model configuration and even the optimizer configuration.This can make us checkpoint the model and then resume the training later from the exact state even without accessing original code.

We can save the entire model using the model.save() method.


```
# Creating a trivial model
modelhere = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(10, activation='softmax')
])

# Compiling the model
modelhere.compile(optimizer='rmsprop',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

# Fitting the model
modelhere.fit(data, labels, batch_size=32, epochs =5)

## Saving the entire model to hdf5 format
modelhere.save('entiremodel.h5')

## Recreating the model using the model that is saved
recreated_entire_model = tf.keras.models.load_model('entiremodel.h5')

recreated_entire_model.summary()
```

    Epoch 1/5
    1000/1000 [==============================] - 0s 304us/sample - loss: 2.3387 - acc: 0.0940
    Epoch 2/5
    1000/1000 [==============================] - 0s 164us/sample - loss: 2.3093 - acc: 0.1070
    Epoch 3/5
    1000/1000 [==============================] - 0s 146us/sample - loss: 2.3029 - acc: 0.1080
    Epoch 4/5
    1000/1000 [==============================] - 0s 129us/sample - loss: 2.2935 - acc: 0.1240
    Epoch 5/5
    1000/1000 [==============================] - 0s 130us/sample - loss: 2.2868 - acc: 0.1340
    Model: "sequential_22"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_76 (Dense)             (None, 64)                2112      
    _________________________________________________________________
    dense_77 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 2,762
    Trainable params: 2,762
    Non-trainable params: 0
    _________________________________________________________________


## Distributed Training


### Estimators

The [Estimators](https://www.tensorflow.org/guide/estimators) api is used to train the tf model in distributed environment. This can be used on the large scale models that are used in production

A normal tf.keras.Model can be trained with tf.estimator API by converting the model to an tf.estimator.Estimator object using [tf.keras.estimator.model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator). We cant use normal model to be trained in distributed environtment.


```
samplemodel = tf.keras.Sequential([
                                   layers.Dense(64, activation='relu', input_shape=(32,)),
                                   layers.Dense(10,activation='softmax')
                                   ])
samplemodel.compile(optimizer = 'rmsprop',
                    loss ='categorical_crossentropy',
                    metrics = ['accuracy'])

sampleestimator = tf.keras.estimator.model_to_estimator(samplemodel)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpntgv4rh_
    INFO:tensorflow:Using the Keras model provided.
    INFO:tensorflow:Using config: {'_model_dir': '/tmp/tmpntgv4rh_', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fbf2758e5c0>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}


### Using Multiple GPUs
tf.keras models can run on multiple GPUs using [tf.contrib.distribute.DistributionStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy). Using this api we can train on multiple GPUs using no extra modification to the code.

Currently, [tf.contrib.distribute.MirroredStrategy]() is the only strategy that is supported in distribution strategy.

What this MirroredStrategy does is in-graph replication with synchronus training using all-reduce in single machine.

To use this distribution strategy with keras we have to convert the model of keras to estimator as mentioned above using the tf.keras.estimators.model_to_estimator method.

The following example will distribute the keras model in to multiple GPUs on a single machine.


```
distributed_model = tf.keras.Sequential([
                    layers.Dense(16, activation='relu', input_shape=(10,)),
                    layers.Dense(1, activation='sigmoid')
                                        ])

distributed_model.compile(optimizer = tf.train.GradientDescentOptimizer(0.001),
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])

distributed_model.summary()
```

    Model: "sequential_24"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_80 (Dense)             (None, 16)                176       
    _________________________________________________________________
    dense_81 (Dense)             (None, 1)                 17        
    =================================================================
    Total params: 193
    Trainable params: 193
    Non-trainable params: 0
    _________________________________________________________________


#### Defining Input Pipeline for distribution
We have to use [tf.data.Dataset]() object to pass the data in form of pipeline to the model where we can distribute the data across multiple GPUs where each device will compute a slice of data


```
def input_pipeline():
    # Each input x is of shape with 10 and there are 1024 examples
    x = np.random.random((1024,10))

    # For each of those example we have y here which is label 0 or 1
    y = np.random.randint(2, size=(1024,1))

    x = tf.cast(x, dtype=tf.float32)
    # Converting normal numpy arrays to tf dataset format
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.repeat(10).batch(32)

    return dataset
```

Now that our data is ready we have to set up the strategy for the model and specify that in some config which is going to used in the tf.keras.estimator.model_to_estimator. The config is required to use the MirroredStrategy which uses all the available GPUs in the machine

But for this 2 steps are required :
- Create [tf.contrib.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/MirroredStrategy) instance

- Create another instance of configuration using [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) and use the above created config instance for the attribute train_distribute inside this


```
# In this strategy we can specify the number of GPUs that are there on
#   EACH WORKER. where each worker is a machine.
sample_strategy = tf.contrib.distribute.MirroredStrategy()
sample_config = tf.estimator.RunConfig(train_distribute= sample_strategy)
```

    INFO:tensorflow:Device is available but not used by distribute strategy: /device:CPU:0
    INFO:tensorflow:Device is available but not used by distribute strategy: /device:XLA_CPU:0
    INFO:tensorflow:Device is available but not used by distribute strategy: /device:XLA_GPU:0
    INFO:tensorflow:Initializing RunConfig with distribution strategies.
    INFO:tensorflow:Not using Distribute Coordinator.


After setting the strategy and config the final step is to conver the normal keras model to estimator using the tf.keras.estimator.model_to_estimator method and inside that we have to specify the required config and model as well as the model directory


```
distributed_estimator = tf.keras.estimator.model_to_estimator(
    keras_model = distributed_model,
    config = sample_config,
    model_dir = '/tmp/modeldir'
)

```

    INFO:tensorflow:Using the Keras model provided.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    INFO:tensorflow:Using config: {'_model_dir': '/tmp/modeldir', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': <tensorflow.contrib.distribute.python.mirrored_strategy.MirroredStrategy object at 0x7fbf241ba080>, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fbf241ba240>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_distribute_coordinator_mode': None}


Now that everything is set all that is requried is to train this model


```
########
### THIS WILL THROW ERROR IF COLAB RUNTIME IS GPU
#######

# Basically when we are using this type of strategy
# we need atleast more than 1 gpu per worker. But Colab
#   provides only one K80 gpu per session.

#######
### THIS WORKS FINE WITH CPU RUNTIME.
#######

distributed_estimator.train(input_fn= input_pipeline,
                            steps = 10)
```

    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:batch_all_reduce: 4 all-reduces with algorithm = nccl,num_packs = 1, agg_small_grads_max_bytes = 0 and agg_small_grads_max_group = 10
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Reduce to /replica:0/task:0/device:CPU:0 then broadcast to ('/replica:0/task:0/device:CPU:0',).
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Warm-starting with WarmStartSettings: WarmStartSettings(ckpt_to_initialize_from='/tmp/modeldir/keras/keras_model.ckpt', vars_to_warm_start='.*', var_name_to_vocab_info={}, var_name_to_prev_var_name={})
    INFO:tensorflow:Warm-starting from: ('/tmp/modeldir/keras/keras_model.ckpt',)
    INFO:tensorflow:Warm-starting variables only in TRAINABLE_VARIABLES.
    INFO:tensorflow:Warm-started 4 variables.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.



    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1355     try:
    -> 1356       return fn(*args)
       1357     except errors.OpError as e:


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _run_fn(feed_dict, fetch_list, target_list, options, run_metadata)
       1338       # Ensure any changes to the graph are reflected in the runtime.
    -> 1339       self._extend_graph()
       1340       return self._call_tf_sessionrun(


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _extend_graph(self)
       1373     with self._graph._session_run_lock():  # pylint: disable=protected-access
    -> 1374       tf_session.ExtendSession(self._session)
       1375 


    InvalidArgumentError: No OpKernel was registered to support Op 'NcclAllReduce' used by {{node training/NcclAllReduce}}with these attrs: [num_devices=1, reduction="sum", shared_name="c0", T=DT_FLOAT]
    Registered devices: [CPU, GPU, XLA_CPU, XLA_GPU]
    Registered kernels:
      <no registered kernels>
    
    	 [[training/NcclAllReduce]]

    
    During handling of the above exception, another exception occurred:


    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-144-ec84a882fefe> in <module>()
          1 distributed_estimator.train(input_fn= input_pipeline,
    ----> 2                             steps = 10)
    

    /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py in train(self, input_fn, hooks, steps, max_steps, saving_listeners)
        365 
        366       saving_listeners = _check_listeners_type(saving_listeners)
    --> 367       loss = self._train_model(input_fn, hooks, saving_listeners)
        368       logging.info('Loss for final step: %s.', loss)
        369       return self


    /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py in _train_model(self, input_fn, hooks, saving_listeners)
       1154   def _train_model(self, input_fn, hooks, saving_listeners):
       1155     if self._train_distribution:
    -> 1156       return self._train_model_distributed(input_fn, hooks, saving_listeners)
       1157     else:
       1158       return self._train_model_default(input_fn, hooks, saving_listeners)


    /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py in _train_model_distributed(self, input_fn, hooks, saving_listeners)
       1217       self._config._train_distribute.configure(self._config.session_config)
       1218       return self._actual_train_model_distributed(
    -> 1219           self._config._train_distribute, input_fn, hooks, saving_listeners)
       1220     # pylint: enable=protected-access
       1221 


    /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py in _actual_train_model_distributed(self, strategy, input_fn, hooks, saving_listeners)
       1327         return self._train_with_estimator_spec(estimator_spec, worker_hooks,
       1328                                                hooks, global_step_tensor,
    -> 1329                                                saving_listeners)
       1330 
       1331   def _train_with_estimator_spec_distributed(self, estimator_spec, worker_hooks,


    /usr/local/lib/python3.6/dist-packages/tensorflow_estimator/python/estimator/estimator.py in _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners)
       1478         save_summaries_steps=save_summary_steps,
       1479         config=self._session_config,
    -> 1480         log_step_count_steps=log_step_count_steps) as mon_sess:
       1481       loss = None
       1482       any_step_done = False


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in MonitoredTrainingSession(master, is_chief, checkpoint_dir, scaffold, hooks, chief_only_hooks, save_checkpoint_secs, save_summaries_steps, save_summaries_secs, config, stop_grace_period_secs, log_step_count_steps, max_wait_secs, save_checkpoint_steps, summary_dir)
        582       session_creator=session_creator,
        583       hooks=all_hooks,
    --> 584       stop_grace_period_secs=stop_grace_period_secs)
        585 
        586 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in __init__(self, session_creator, hooks, stop_grace_period_secs)
       1005         hooks,
       1006         should_recover=True,
    -> 1007         stop_grace_period_secs=stop_grace_period_secs)
       1008 
       1009 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in __init__(self, session_creator, hooks, should_recover, stop_grace_period_secs)
        723         stop_grace_period_secs=stop_grace_period_secs)
        724     if should_recover:
    --> 725       self._sess = _RecoverableSession(self._coordinated_creator)
        726     else:
        727       self._sess = self._coordinated_creator.create_session()


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in __init__(self, sess_creator)
       1198     """
       1199     self._sess_creator = sess_creator
    -> 1200     _WrappedSession.__init__(self, self._create_session())
       1201 
       1202   def _create_session(self):


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in _create_session(self)
       1203     while True:
       1204       try:
    -> 1205         return self._sess_creator.create_session()
       1206       except _PREEMPTION_ERRORS as e:
       1207         logging.info(


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in create_session(self)
        869       """Creates a coordinated session."""
        870       # Keep the tf_sess for unit testing.
    --> 871       self.tf_sess = self._session_creator.create_session()
        872       # We don't want coordinator to suppress any exception.
        873       self.coord = coordinator.Coordinator(clean_stop_exception_types=[])


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/monitored_session.py in create_session(self)
        645         init_op=self._scaffold.init_op,
        646         init_feed_dict=self._scaffold.init_feed_dict,
    --> 647         init_fn=self._scaffold.init_fn)
        648 
        649 


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/session_manager.py in prepare_session(self, master, init_op, saver, checkpoint_dir, checkpoint_filename_with_path, wait_for_checkpoint, max_wait_secs, config, init_feed_dict, init_fn)
        294                            "init_fn or local_init_op was given")
        295       if init_op is not None:
    --> 296         sess.run(init_op, feed_dict=init_feed_dict)
        297       if init_fn:
        298         init_fn(sess)


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
        948     try:
        949       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 950                          run_metadata_ptr)
        951       if run_metadata:
        952         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1171     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1172       results = self._do_run(handle, final_targets, final_fetches,
    -> 1173                              feed_dict_tensor, options, run_metadata)
       1174     else:
       1175       results = []


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1348     if handle is None:
       1349       return self._do_call(_run_fn, feeds, fetches, targets, options,
    -> 1350                            run_metadata)
       1351     else:
       1352       return self._do_call(_prun_fn, handle, feeds, fetches)


    /usr/local/lib/python3.6/dist-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1368           pass
       1369       message = error_interpolation.interpolate(message, self._graph)
    -> 1370       raise type(e)(node_def, op, message)
       1371 
       1372   def _extend_graph(self):


    InvalidArgumentError: No OpKernel was registered to support Op 'NcclAllReduce' used by node training/NcclAllReduce (defined at <ipython-input-144-ec84a882fefe>:2) with these attrs: [num_devices=1, reduction="sum", shared_name="c0", T=DT_FLOAT]
    Registered devices: [CPU, GPU, XLA_CPU, XLA_GPU]
    Registered kernels:
      <no registered kernels>
    
    	 [[training/NcclAllReduce]]


# Eager Execution


Tensorflows eager execution is something that evaluates the tensor object value without building graphs . This will run the operation and returns a value instead of waiting for a session.
This makes the debugging and prototyping of the tensorflow models easier.

Eager execution supports most the Tensorflow operations and also the GPU as well. To know more about these examples we can following the following link [LINK for Examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples)


## Setup and Basic Usage

To start with eager execution all we need to do is use this piece of code at the start tf.enable_eager_execution() at the beginning of the code or the session.

The benifits of using eager execution
- We can structure our code naturally. Also we can iterate and build small models which can be sticthed together

- We can call the operations directly and inspect the model to get the output when we change anything.We can also use standard python tools for debugging purposes

- lets us use normal python control flow instead of graph control flow


```
#@title Default title text
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
tf.__version__


```

    TensorFlow 2.x selected.





    '2.0.0-rc2'



In tensorflow version 2.0 eager execution is enabled by default. Only in case of verion 1.14 we need to enable the eager execution mode by usin the command below

``` python
tf.enable_eager_execution()
```


```
# To check whether these eager execution is enabled or not we can use this
tf.executing_eagerly()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-ed9c80fe6ed3> in <module>()
    ----> 1 tf.executing_eagerly()
    

    NameError: name 'tf' is not defined



```
# Adding a matrix type value here
somevar = [[2.]]
anothervar = tf.matmul(somevar,somevar)
print(anothervar)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-b6a8cd2a88f8> in <module>()
          1 somevar = [[2.]]
    ----> 2 anothervar = tf.matmul(somevar,somevar)
          3 print(anothervar)


    NameError: name 'tf' is not defined


Enabling eager execution will not use graph model now and wont have place holders to send values with feed dict. Instead now what happens is the values are stored concretly and they are evaluated immediately. So we can debug them and print them using the print function.
Evaluating, printing or debugging will not break the flow of computing gradients.

Also this eager execution also works with Numpy. Numpy operations now will accept tf.Tensor arguments. We have tensorflow math operations which will convert normal python objects and numpy arrays to tf.Tensor objects.

REMEMBER THAT tf.Tensor.numpy method will return the objects values as Numpy array


```
# Example demonstrating the tensorflow objects being accepted and
#   evaluated using numpy methods
a = tf.constant([
                 [1,2],
                 [3,34]
                ])
b = tf.add(a,1)

print(a)
print(b)

# Now lets apply a numpy operation using these two a and b
import numpy as np
c = a*b
print(" c value without using numpy and normal eager execution is")
print(c)

## Now using the numpy function and using multiply method
d = np.multiply(a, b)
print("The value of d is , "+str(d))
```


```
# We can also obtain the value of tensor and show it in numpy format
print(a.numpy())
```

Since we dont have graph like structure anymore if we want to use graphs we can make use of the module tf.contrib.eager. That contains symbols which are available to both eager and graph execution environment.


```
tfe = tf.contrib.eager
```

## Dynamic Control flow

Another major benifit of eager execution is that we can make use of any python code and all its functionality is available. A simple fizzbuzz code will demonstrate that.


```
def fizzbuzz(max_num):
    counter = tf.constant(1)
    # Lets declare the maxnum and make it in tensorflow format
    maxnum_tf = tf.convert_to_tensor(max_num)

    while counter<maxnum_tf:
        if int(counter%3)==0 and int(counter%5)==0:
            print('Fizzbuzz')
        elif int(counter%3)==0:
            print("fizz")
        elif int(counter%5)==0:
            print("buzz")
        else:
            print(counter.numpy())
        counter = counter +1
```


```
fizzbuzz(16)
```

## Building a Model
Most of the machine learning models are made up of layers. When using eager execution we can write our own layers or use tf.keras.layers package to create them.

Also tensorflow has tf.keras.layers.Layer which can be used to subclass a model and create a custom one.
Inherit all the properties from it and we can implement our own layer


```
class MySimpleLayer(tf.keras.layers.Layer):
    
    def __init__(self,output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units
    
    def build(self, input_shape):
        # This build method is called the first time we are using to 
        # create this layer.Here we can create variables where their 
        # shapes can be derived from the input_shape. So it removes the extra
        # need for the user to specify full shapes. If we know the full shapes
        #   we can directly create them in th init itself

        # Here we are creating a variable with name kernel and scope as well
        # With the shape as last value of input_shape and the output units
        self.kernel = self.add_variable("kernel", 
                                        [input_shape[-1], self.ouput_units])
    
    def call(self,input):
        # Over riding call() instead of __call__ so we can modify something
        # call() is made first and is called inside __call__ , rhymes isn't it?
        return tf.matmul(input, self.kernel)
```


```
class MySimpleDenseLayer(tf.keras.layers.Dense):

    def __init__(self, output_units):
        super(MySimpleDenseLayer, self).__init__()
        self.output_units = output_units
    
    def build(self, input_units):
        # Building something , like variables that cna be used further
        self.kernel = self.add_variable('kernel',
                                        [input_shape[-1], self.output_units])
    
    def call(self, input):
        return tf.matmul(input, self.kernel)

```

When composing layers in to models we can use tf.keras.Sequential to represent models which are linear stack of layers.


```
model = tf.keras.Sequential([MySimpleLayer(10),
                             tf.keras.layers.Dense(10)])
```

Or else we can organize models in to classes by inherting from tf.keras.Model. This is a container for layers and this by itself is another layer, where it allows tf.keras.Model objects in to another tf.keras.Model objects


```
class MNISTmodel(tf.keras.Model):
    
    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result

model = MNISTmodel()
```

It is not required to setup input shape for tf.keras.Model class since the parameters 
are set the first time input is passed to the layer.

[tf.keras.layers]() classes create and contain their own model variables that are tied to their lifetime of their layer objects . To share layer variables , share their objects

So we instantiate the objects and then use that to share them

## Eager Training


### Computing the Gradients (Differentiation of variables)

[Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) , which basically means that we apply chain rule to the variables and evaluate the derivatives until that rule is propagated and then split them and evaluate the derivatives. This is also useful to propagate the backpropagation of the variables for training neural networks

We can use [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) to train and evaluate the gradients in eager mode. We can use this in complicated training loops where we loop over the batch dataset in tensorflow format.

Since the differentiation occurs in each call of the epoch we record the gradients in a series wise format. To compute the gradients we play the tape which is recorded and then discard it. It can only be used once on a variable through a particular runtime , running it again will throw error as the gradients are already propagated.



```
w = tf.Variable(name='w',initial_value=[1.0])
```


```
print(w)
```


```
with tf.GradientTape() as tape:
    loss = w*w

# To compute loss of function f(x,y) wrto x we write the
#      tape as like this tape.gradient(f(x,y), x)
#   Which returns the gradietns to us
grad_loss = tape.gradient(loss, w)
print("The gradient of the loss is "+str(grad_loss))
print(grad_loss.numpy())
```

### Trainig a model , Any keras one 


Below example will create a multi layer model which classifies the standard MNIST images which are handwritten. It demonstratest the optimizer and layer APi to build trainable graphs in eager execution environment


```
(mnist_images, mnist_labels),_ = tf.keras.datasets.mnist.load_data()

## ... are called ellipsis and they are used to specify any unspecified
#   dimensions inthe array while slicing it

mnist_images = tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32)
mnist_labels = tf.cast(mnist_labels, tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((mnist_images, mnist_labels))

dataset = dataset.shuffle(1000).batch(32)
```


```
# Building the model
mnist_model = tf.keras.Sequential([
                                   tf.keras.layers.Conv2D(16, [3,3], activation='relu',
                                                          input_shape = (None, None, 1)),
                                   tf.keras.layers.Conv2D(16, [3,3], activation='relu'),
                                   tf.keras.layers.GlobalAveragePooling2D(),
                                   tf.keras.layers.Dense(10)
                                ])

```

While the keras model has inbuilt fit method to train sometimes when we need more customization we can implement training loop using eager method as follows.


```
for images,labels in dataset.take(1):
    print("Logits :" , mnist_model( images[0:1]).numpy() )
```


```
optimizer = tf.keras.optimizers.Adam()

# WE use sparse categorical cross entropy when we have multiple labels
# If they are specified in interger format this is used mostly. IF that is
#    is in one hot encoded vector then we have to use normal categorical
#   cross entropy
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

loss_history = []
```


```
def train_step(images, labels):

    # Activating the tape here and calculating the loss for these bunch of images
    with tf.GradientTape() as tape:
        # Calculating the predictions for images
        logits = mnist_model(images, training = True)
        tf.debugging.assert_equal(logits.shape, (32, 10))

        # Calculating the loss from these predictions
        # Remember that we have to give actual labels in first argument
        #   and the predictions in the second argument
        loss_here = loss_object(labels, logits)
    
    # Appending the mean loss and storing this in this loss history array
    loss_history.append(loss_here.numpy().mean())

    # Applying the tape to calculate the derivates of model variables
    #   wrto the loss of current images
    grads = tape.gradient(loss_here, mnist_model.trainable_variables)

    # Using those gradients to update the weights using some optimizer
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
```


```
def train():
    # Running the training for given number of epochs
    for epoch in range(3):

        # And in each epoch we are enumerating over dataset
        #   And then the batched dataset is being used for each
        #   training_step
        for (batch, (images, labels)) in enumerate(dataset):
            
            # For each batch size we are updating variables using
            #   optimizer we have created
            train_step(images,labels)
        print("Epoch {} is finished".format(epoch))
```


```
train()
```


```
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
```

### Variables and Optimizers

`tf.Variable` objects store a mutable `tf.Tensor` values that are accesed during training to make automatic differentiation easier with `tf.GradientTape` . The parameters of the model can be encapsulated in classes or variables.

We can encapsulate model parameters by using tf.Variable on tf.GradientTape.

Simple model example on how to use variables in any model to train and update the variables.


```
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        # Define the float values for the variables
        # It will throw error if these inputs are integer
        #   As it for some reason computes only multiplication
        #   for float tensors.
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name = "bias")
    
    def call(self, inputs):
        return inputs * self.W + self.B
    
# Creating a small dataset 
num_examples = 2000
train_inputs = tf.random.normal([num_examples])
noise = tf.random.normal([num_examples])
train_outputs = train_inputs * 3 + 2 + noise

# Defining a loss function that needs to be optimized
# We are using mean squared error.
def loss(model, inputs, targets):
    preds = model(inputs)
    error = preds-targets
    squarederror = tf.square(error)
    meansquarederror = tf.reduce_mean(squarederror)
    return meansquarederror

# Gradient calculating function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, model.trainable_variables)


## Defining Model
# Derivates of the loss function with respect to variables
# And optimizer strategy to updating that respective variables
model = Model()
optimizer = tf.keras.optimizers.Adam(0.01)

####
# Usage of {} in strings basically usage of format()
# To know more about formatting strings google it
# Basically {} is used to denote the variables that are passed to string
#   Inside {} it takes the form of {a:b} where a is the index sort of thing
#   and b specify the properties of the integer.

print("The intial loss, {:.3f} ".format(loss(model, train_inputs, train_outputs)))

########
### Training Loop
########
for i in range(3000):
    grads = grad(model, train_inputs, train_outputs)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if i%20 == 0:
        print("loss at step {:03d} {:.3f}".format(i, loss(model, train_inputs, train_outputs)))
    
print("Final loss is {:.3f}".format(loss(model, train_inputs, train_outputs)))
print("W is {} and b is {}".format(model.W.numpy(), model.B.numpy()))
```

## Using Objects for state during the eager execution


### Variables are Objects in Eager


With tf graph execution the variables are saved to the session object . But in case of eager execution they are tied to the object that is instantiated. So the value of the variables are tied to the python objects



```
if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device('gpu:0'):
        print("GPU is enabled now and the device is 0")

        # Now the variable is stored in memory and takes up
        #   GPU computatinoal space
        v = tf.Variable(tf.random.normal([1000,1000]))
        
        # Now the variable is reassigned so the gpu no longer takes
        #   up that space
        v = None
```

### Object based saving Checkpoints


This section covers small details of the chapter [Checkpoints guide](https://www.tensorflow.org/guide/checkpoint).
The tf.train.Checkpoint can save the variables tf.Variable states and write it to ckpt files . Also we can restore the variables value


```
# Assinging values to the variable x 
x = tf.Variable(10.0, name = 'x')

# Creating an instance of checkpoint
checkpoint = tf.train.Checkpoint(x=x)


# Now before assignign the variabel value of x it is 10
# Lets reassign valeu of x to 2 and then save it

# Remember that when using assign function the dtypes of the 
#   variable that we are using should match, The value here 2.0 is float
x.assign(2.)

# Creating a checkpiont name, in the file object that will be suffixed
checkpoint_path = './ckpt'
checkpoint.save('./ckpt')
```

After the value of x is assigned we are using checkpoint path as a name to save the checkpoint files that contains the current information about x. We can restore that information of the checkpoint using the restore method of the checkpoint object we have instantiated earlier. After restoring the checkpoint using the argumetn as `tf.train.latest_checkpoint(path)` we can print the value of x and it is the value when we saved it.


```
# We are changing the variable after assiging it
x.assign(11)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
print("The valeu of x is {}".format(x.numpy()))
```

What is the point of saving just variables when we want to save the whole model. So we generally save the full keras model as objects that can be restored even if we dont have the code that created the model itself.
To record the state of the model, its optimizer and global step. We can pass them to tf.train.checkpoint. With the arguments of model and optimizer we can specify what to save and when to save by using this inside a loop. So that we can have weights or model snapshots at series of time during training.


```
import os

samplemodel = tf.keras.Sequential([
                                   tf.keras.layers.Conv2D(16, [3,3], activation = 'relu'),
                                   tf.keras.layers.GlobalAveragePooling2D(),
                                   tf.keras.layers.Dense(10)
                                 ])
optimizer = tf.train.AdamOptimizer(0.01)
checkpoint_dir = './samplemodel'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Creating path for saving the checkpoint.
# We already created a folder and the name for the files
#    will be sampleckpt 
checkpoint_prefix = os.path.join(checkpoint_dir, 'sampleckpt')

# Instantiating Checkpoint object
checkpoint_object = tf.train.Checkpoint(model = samplemodel,
                               optimizer = optimizer)

# Saving the full data here , for the model and optimizer 
#   Using the name sampleckpt in the samplemodel directory
checkpoint_object.save(checkpoint_prefix)

# Restoring the model value and its parameters using this following function
checkpoint_object.restore(tf.train.latest_checkpoint(checkpoint_prefix))

```

### Object Oriented Metrics


`tf.keras.metrics` are store as objects. We can update the key metric by passing the callable objects and retreive the new result using the `tf.keras.metric.result` method.


```
x = tf.keras.metrics.Mean('loss')
x(0)
x(4)

# This will give us resut of 2
x.result()

# Now we are adding more numbers
# Considering the previous two as well
#   The average of 0,4,3,4,5 which is 3.2
x([3,4,5])
x.result()
```

### Summaries and Tensorboard

Tensorboard is a visualization tool that helps us keeping track of variables that we can name in the scopes. It uses summary events that are written while executing the programs

We can use tf.Summary() to record the variables in the eager execution mode. For example, to record the summary once every 100 training steps while the training is going on, we can use the following


```
logdir = './tbdata'

# Now the contrib module contains some of the fucntionailty of 
#   summary here
writer = tf.summary.create_file_writer(logdir)

with writer.as_default():
    for i in range(1000):
        step = i+1
        loss = 1 - 0.001*step

        if step % 100 == 0:
            tf.summary.scalar('loss', loss, step = step)
```


```
!ls tbdata
```

## Advanced Automatic differentiation


#### Dynamic models
`tf.GradientTape()` can be used in dynamic models. This example of backtracking line search algorithm looks sort of like Numpy code, except there are gradients and is differentiable, despite the complex control flow


```
def line_search_step(fn, init_x, rate = 1.0):
    with tf.GradientTape() as tape:
        # Variables are automatically recorded
        tape.watch(init_x)
        value = fn(init_x)
    grad = tape.gradient(value, init_x)
    grad_norm = tf.reduce_sum(grad*grad)
    init_value = value

    while value > init_value - rate*grad_norm:
        x = init_x - rate*grad
        value = fn(x)
        rate /= 2.0
    return x. value
```


```

```


```

```
