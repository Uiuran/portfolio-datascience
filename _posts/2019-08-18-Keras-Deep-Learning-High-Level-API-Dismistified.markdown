---
layout: post
title:  "Hacking the Most Popular High Level Deep Learning API: PART I"
categories: keras tensorflow deeplearning
comments: true
published: true
---
# [Keras Internals: PART I - Crash Overview][kerasurl]


Although [Google's TensorFlow][tensorflowurl] is recognized between [one of many][dllibs], it's python High Level API Keras still the most popular python module to products and research in Deep Neural Networks. It emboddies simplicity in the use and allows the user fast access to the elements AI/Deep Neural Networks Research, paving the way to the popularization of Neural Network modelling.

This post is an intended guide to hack/uncover python APIs designed for high level scientific computation built up on diverse computing paradigm, in this case the Computational Graph language paradigm.

## Keras File Tree
|![keras file tree](/assets/kerasfiletree.png)|
| *Keras File Tree is an example of well organized Open Source Python Software based on another lib* |

In the representation given above we purposely surpress some of the Keras modules parts such: datasets, preprocessing and applications. A brief overview inside the keras folders evidence the importance of **backend**, **engine** and **layers** modules for Keras and model building.

### [Keras Backend: Where it touches TensorFlow's(and others) Backend][kbackend]
Dont expect to find TensorFlow code in Keras Module other then the Backend, after all this is why it has the name, it has all to do with the organization of the software in its modules:
- load_backend.py uses backend() to load file tensorflow_backend.py imports all that is needed from low-level TensorFlow python API. From here you already know, that is Tensorflow Low-Level, below this will gonna get te cpp and so on (future post...).
- Wont spect any other TensorFlow code scattered somewhere else, all the rest of the code is the works of Keras API.
  
 ### [Keras Engine: Networks, Graph Nodes and Layers][kengine]
Here we find the soul of the Keras API: 
- Layers are the abstraction for trainable objects in a Neural Network, receiving inputs and outputs. It is implemented as specific types of TensorFlow Network parts in the specific layers modules such convolutions (Conv2D parts and so on).
- Nodes are the structures behind the graphs constructions, each node has a Layer and a bunch of input layers and output layers. So dont confuse Node by a neuron inside a specific Layer to be built in the Comp Graph, such a Conv2D Layer. Rather a Node will be the overall Name Scope composing all the TensorFlow Operations necessary to fully perform a Convolutions Neural Network Layer step(what Keras does is to easy this building of the Graph by defining the Node with appropriated Convolutional Layers operations).

One will find these basic units definitions at the engine [base_layers.py][kbase], Layer:

```python
class Layer(object):
    """Abstract base layer class.
    # Properties
        input, output: Input/output tensor(s). Note that if the layer
            is used more than once (shared layer), this is ill-defined
            and will raise an exception. In such cases, use
            `layer.get_input_at(node_index)`.
        input_mask, output_mask: Mask tensors. Same caveats apply as
            input, output.
        input_shape: Shape tuple. Provided for convenience, but note
            that there may be cases in which this attribute is
            ill-defined (e.g. a shared layer with multiple input
            shapes), in which case requesting `input_shape` will raise
            an Exception. Prefer using
            `layer.get_input_shape_at(node_index)`.
        input_spec: List of InputSpec class instances
            each entry describes one required input:
                - ndim
                - dtype
            A layer with `n` input tensors must have
            an `input_spec` of length `n`.
        name: String, must be unique within a model.
        non_trainable_weights: List of variables.
        output_shape: Shape tuple. See `input_shape`.
        stateful: Boolean indicating whether the layer carries
            additional non-weight state. Used in, for instance, RNN
            cells to carry information between batches.
        supports_masking: Boolean indicator of whether the layer
            supports masking, typically for unused timesteps in a
            sequence.
        trainable: Boolean, whether the layer weights
            will be updated during training.
        trainable_weights: List of variables.
        uses_learning_phase: Whether any operation
            of the layer uses `K.in_training_phase()`
            or `K.in_test_phase()`.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        dtype:  Default dtype of the layers's weights.
```
and for the Node:

```python

class Node(object):
    """A `Node` describes the connectivity between two layers.
    Each time a layer is connected to some new input,
    a node is added to `layer._inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer._outbound_nodes`.
    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.
    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:
    origin_node = inbound_layers[i]._inbound_nodes[node_indices[i]]
    input_tensors[i] == origin_node.output_tensors[tensor_indices[i]]
    A node from layer A to layer B is added to:
        A._outbound_nodes
        B._inbound_nodes
    """
```
- Finally, the [Networks][knetworks] are the connectomes of Nodes and Layers, being  each Node composed by Layers that will be the specifications of unit architectures math, mapped to TensorFlow operations (as already mentioned above).

As one can see, the network just builds the Comp Graphs (emulating the TensorFlow Low Level API of Graph Building)

```python
class Network(Layer):
    """A Network is a directed acyclic graph of layers.
    It is the topological form of a "model". A Model
    is simply a Network with added training routines.
    # Properties
        name
        inputs
        outputs
        layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        dtype
        input_shape
        output_shape
        weights (list of variables)
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        losses
        updates
        state_updates
        stateful
    # Methods
        __call__
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape
        save
        add_loss
        add_update
        get_losses_for
        get_updates_for
        to_json
        to_yaml
        reset_states
    # Class Methods
        from_config
    # Raises
        TypeError: if input tensors are not Keras tensors
            (tensors returned by `Input`).
    """
    
```
   and the graph building function from network.py 

```python   

145 def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
...
``` 
### [Keras Layers: Implement Deep Neural Nets Specific Layered Steps][klayers]

Since we gotta the TensorFlow operations with the **Backend** and engineered each node part to build the Computational Graph in **Engine**, the specification of the Nodes composition must be given, and it is done as it is in [Conv2D][https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py]. 

In this stage of the Keras software architecture, using inheritance from Layers as abstracts of Model Parts allows to modular expansion of the API with backward compatibility, and using wrappers such ```_Conv2D``` allows to automaticly port legacy code, that is intended to be mainteined since its predict to still be used by alot of modellers. 


## Concluding Remarks

- Keras is a small port, but still very well architected and possibly scalable for a big size AI Machine Learning API. Separating software architecures that deals with backends from the main engine abstractions, templates and specific implementations. 
- The division in the architecture, allows for easy debugging as backends changes its versions, possibly growing capabilities, without crashing other parts of the software. Also, it is easy to grow specific math modelling features.
- Keras follows the pace and the model of TensorFlow APIs development, always seeking for backward compatibility, use of similar features such decorators, legacy code handling and wrappers for specifying lib names.

## Next Posts

- In the next parts i plan to detail some of the relevant implementation of the **Engine**, mainly concerning Graph Building. I plan to present some examples on how to hack the API such that you can expand and contribute to Keras too. 
- More on the supplementary parts of the API will be given as seems to be needed for the understanding of the API.

*** Focusing on TensorFlow

[kerasurl]: https://github.com/keras-team/keras 
[tensorflowurl]: https://github.com/tensorflow
[dllibs]: https://en.wikipedia.org/wiki/Comparison_of_deep-learning_software#Deep-learning_software_by_name
[kbackend]: https://github.com/keras-team/keras/tree/master/keras/backend
[tfbackend]: https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
[kengine]: https://github.com/keras-team/keras/blob/master/keras/engine
[kbase]: https://github.com/keras-team/keras/blob/master/keras/engine/base_layer.py
[knetworks]: https://github.com/keras-team/keras/blob/master/keras/engine/network.py
[klayers]: https://github.com/keras-team/keras/blob/master/keras/layers
[convolutions]: https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py
