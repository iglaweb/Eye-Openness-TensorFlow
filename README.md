# Eye-Openness-TensorFlow

# Overview

The eye state classifier is based on transfer learning technique. I use transfer learning where we take a pre-trained network (trained on about a million general images), use it to extract features, and train a new layer on top for our own task of classifying images of eyes.

I used a Model Maker library that simplifies the process of adapting and converting a TensorFlow neural-network model to particular input data. The newly created model is based on MobileNetV2 backbone.

I provide full training code, data preparation scripts, and a pretrained model.

The detector has speed **~10-15 ms/image** for floating model (image size is 224x224, Macbook Pro, CPU, 2 GHz Quad-Core Intel Core i5).


## How to use the pretrained model

To use the pretrained model you will need to download `run_inference.py` and  
a tensorflow lite model (`.tflite` file, it is [here](https://drive.google.com/drive/folders/1oZTsJ550O-z3ImMlgjWRBo9JhM9vOo6-?usp=sharing)). You can see an example of usage in `DrowinessTraining.ipynb`. 


## Image Dataset
The model was trained using image dataset [mrlEyes_2018_01](http://mrl.cs.vsb.cz/eyedataset).



## Requirements

* tensorflow 2.+ (inference was tested using tensorflow 2.4.0-dev20200810)
* tflite-model-maker (training was tested with 0.2.0)
* opencv-python


## Issues

If you find any problems or would like to suggest a feature, please
feel free to file an [issue](https://github.com/iglaweb/Eye-Openness-TensorFlow/issues)

## License

    Copyright 2020 Igor Lashkov

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.