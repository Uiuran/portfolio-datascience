---
layout: post
title:  "Hacking the Most Popular High Level Deep Learning API: PART II"
categories: keras tensorflow deeplearning
comments: true
published: false
---

# [Keras Internals: PART II - The Engine][kerasengine]
Keras coming with TensorFlow core is essentialy some steps ahead the last Keras from  Keras-Team. 
The reason for this difference is, probably, the rithmin of TensorFlow core development is much more rapid than its corresponding High Level API, this is also the main reason on Why Keras is developed at the Core of TensorFlow.
In Part I i gave an overview of the Keras developed by Keras-Team in version 2.2.4, it's structure remains very similar in the Keras 2.3.0 of the TensorFlow 2.0beta Core, however they added a lot more of relevant things, lets take a look in these diffs first.

## Overview of Keras 2.2.4 and Keras TF 2.0beta Core 2.3.0 File Trees

![keras file tree](/assets/kerasfiletree.png)
|*Keras File Tree is an example of well organized Open Source Python Software based on another lib*|
  
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
[tensorflowurl]: https://github.com/tensorflow
[dllibs]: https://en.wikipedia.org/wiki/Comparison_of_deep-learning_software#Deep-learning_software_by_name
[kbackend]: https://github.com/keras-team/keras/tree/master/keras/backend
[tfbackend]: https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
[kengine]: https://github.com/keras-team/keras/blob/master/keras/engine
[kbase]: https://github.com/keras-team/keras/blob/master/keras/engine/base_layer.py
[knetworks]: https://github.com/keras-team/keras/blob/master/keras/engine/network.py
[klayers]: https://github.com/keras-team/keras/blob/master/keras/layers
[convolutions]: https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py
