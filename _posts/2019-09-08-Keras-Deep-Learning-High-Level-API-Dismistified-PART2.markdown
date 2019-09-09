---
layout: post
title:  "Hacking the Most Popular High Level Deep Learning API: PART II"
categories: keras tensorflow deeplearning
comments: true
published: false
---

# Keras Internals: PART II - The Engine
Keras coming with TensorFlow core is essentialy some steps ahead the last Keras from  Keras-Team. 
The reason for this difference is, probably, the rithmin of TensorFlow core development is much more rapid than its corresponding High Level API, this is also the main reason on Why Keras is developed at the Core of TensorFlow.

In [Part I][partI] i gave an overview of the main classes structuring Keras developed by Keras-Team in version 2.2.4. It does remains very similar to Keras 2.3.0 of the TensorFlow 2.0beta Core, however they added a lot more of relevant things.

We will first take a look in these differences, starting from __init__.py, them studying the structures of the three class presented so far: Node, Layers, Network

## Overview of [Keras 2.2.4][kerasengine] and [Keras TF 2.0beta Core 2.3.0][tfkerasengine] File Trees

![keras file tree](/assets/engines.png)
|*Keras 2.2.4 and Keras TF 2.0beta Core 2.3.0 File Trees*|

### __init__.py file

![init](/assets/__init__.png)
|*Keras 2.2.4 and Keras TF 2.0beta Core 2.3.0 File Trees*|
  
### [Keras Engine: Networks, Graph Nodes and Layers][kengine]

Here we find the soul of the Keras API: 
- Layers are the abstraction for trainable objects in a Neural Network, receiving inputs and outputs. It is implemented as specific types of TensorFlow Network parts in the specific layers modules such convolutions (Conv2D parts and so on).
- Nodes are the structures behind the graphs constructions, each node has a Layer and a bunch of input layers and output layers. So dont confuse Node by a neuron inside a specific Layer to be built in the Comp Graph, such a Conv2D Layer. Rather a Node will be the overall Name Scope composing all the TensorFlow Operations necessary to fully perform a Convolutions Neural Network Layer step(what Keras does is to easy this building of the Graph by defining the Node with appropriated Convolutional Layers operations).

```python   

``` 
### [Keras Layers: Implement Deep Neural Nets Specific Layered Steps][klayers]


## Concluding Remarks

## Next Posts

[kerasengine]: https://github.com/keras-team/keras/tree/master/keras/engine 
[tfkerasengine]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/engine
[partI]:https://uiuran.github.io/keras/tensorflow/deeplearning/2019/08/18/Keras-Deep-Learning-High-Level-API-Dismistified.html
[dllibs]: https://en.wikipedia.org/wiki/Comparison_of_deep-learning_software#Deep-learning_software_by_name
[kbackend]: https://github.com/keras-team/keras/tree/master/keras/backend
[tfbackend]: https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
[kengine]: https://github.com/keras-team/keras/blob/master/keras/engine
[kbase]: https://github.com/keras-team/keras/blob/master/keras/engine/base_layer.py
[knetworks]: https://github.com/keras-team/keras/blob/master/keras/engine/network.py
[klayers]: https://github.com/keras-team/keras/blob/master/keras/layers
[convolutions]: https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py
