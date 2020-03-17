---
layout: post
title:  "GANs by Examples - Part 1"
categories: deeplearning nips2016 gan goodfellow
comments: true
published: false
---
## U-Net Architecture for image segmentation
![unet architecture](/assets/unet.jpg)
|*Unet Architecture*|

![unet architecture code](/assets/unet.png)
|*Unet code*|

## ResNet Architecture 

![resnet architecture](/assets/resnet.png)
|*Resnet Architecture*|

![resnet architecture](/assets/Resnetcode.png)
|*Resnet Code*|


Although the network can be translated directly from its architecture diagram, the downsampling and upsampling layers uses instance normalization, thats equivalent to standartize each example in the bach in each channel separately. He (the author of Generative Deep Learning) also pads the image before doing the sampling process. These steps are not directly part of ResNet but maybe the desired answer cant be achieve without the padding because the size wont match and accuracy in the case of instance normalization.

![resnet architecture](/assets/resnetcode1.png)

|*Resnet Code 2*|

[goodfellownips2016]: https://www.youtube.com/watch?v=HGYYEUSm-0Q
[goodfellownips2016paper]: https://arxiv.org/pdf/1701.00160.pdf
[goodfellowpodcast]: https://www.youtube.com/watch?v=Z6rxFNMGdn0
