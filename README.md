
<p align="center">
  <H2>Memory Management for the GPU Poor by DeepBeepMeep</H2>	
</p>


This module contains multiples optimisations so that models such as Flux (and derived), Mochi, CogView, HunyuanVideo, ...  can run smoothly on a 24 GB GPU limited card. 
This a replacement for the accelerate library that should in theory manage offloading, but doesn't work properly with models that are loaded / unloaded several
times in a pipe (eg VAE).

Requirements:
- GPU: RTX 3090/ RTX 4090 (24 GB of VRAM)
- RAM: minimum 48 GB, recommended 64 GB 

## Usage 
First you need to install the module in your current project with:
```shell
pip install mmgp
```

It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
1) First make sure that the pipeline explictly loads the models in the CPU device, for instance:
```
  pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
```

2) Once every potential Lora has been loaded and merged, add the following lines:

```
  from mmgp import offload
  offload.all(pipe)
```  

## Options
The 'transformer' model in the pipe contains usually the video or image generator is quantized on the fly by default to 8 bits. If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. In that case you need to set the option *quantizeTransformer* to *False* to turn off on the fly quantization.

You can specify a list of additional models string ids to quantize (for instance the text_encoder) using the optional argument *modelsToQuantize* for instance *modelsToQuantize = ["text_encoder_2"]*.This may be useful if you have less than 48 GB of RAM.

Note that there is little advantage on the GPU / VRAM side to quantize text encoders as their inputs are usually quite light. 

Conversely if you have more than 64GB of RAM you may want to enable RAM pinning with the option *pinInRAM = True*. You will get in return super fast loading / unloading of models
(this can save significant time if the same pipeline is run multiple times in a row)

In Summary, if you have:
- Between 32 GB and 48 GB of RAM
```
  offload.all(pipe, modelsToQuantize = ["text_encoder_2"]) # for Flux models
  #OR
  offload.all(pipe, modelsToQuantize = ["text_encoder"]) # for HunyuanVideo models

```  

- Between 48 GB and 64 GB of RAM
```
  offload.all(pipe)
```  
- More than 64 GB of RAM
```
  offload.all(pipe), pinInRAM = True
```

## Special
Sometime there isn't an explicit pipe object as each submodel is loaded separately in the main app. If this is the case, you need to create a dictionary that manually maps all the models.\
For instance :


- for flux derived models: 
```
pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
```
- for mochi: 
```
pipe = { "text_encoder": self.text_encoder, "transformer": self.dit, "vae":self.decoder }
```


Please note that there should be always one model whose Id is 'transformer'. It corresponds to the main image / video model which usually needs to be quantized (this is done on the fly by default when loading the model).

Becareful, lots of models use the T5 XXL as a text encoder. However, quite often their corresponding pipeline configurations point at the official Google T5 XXL repository 
where there is a huge 40GB model to download and load. It is cumbersorme as it is a 32 bits model and contains the decoder part of T5 that is not used. 
I suggest you use instead one of the 16 bits encoder only version available around, for instance:
```
text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=torch.float16)
```

Sometime just providing the pipe won't be sufficient as you will need to change the content of the core model: 
- For instance you may need to disable an existing CPU offload logic that already exists (such as manual calls to move tensors between cuda and the cpu)
- mmpg to tries to fake the device as being "cuda" but sometimes some code won't be fooled and it will create tensors in the cpu device and this may cause some issues.

You are free to use my module for non commercial use as long you give me proper credits. You may contact me on twitter @deepbeepmeep

Thanks to
---------
- Huggingface / accelerate for the hooking examples
- Huggingface / quanto for their very useful quantizer
- gau-nernst for his Pinnig RAM samples