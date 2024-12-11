# ------------------ Memory Management for the GPU Poor by DeepBeepMeep (mmgp)------------------
#
# This module contains multiples optimisations so that models such as Flux (and derived), Mochi, CogView, HunyuanVideo, ...  can run smoothly on a 24 GB GPU limited card. 
# This a replacement for the accelerate library that should in theory manage offloading, but doesn't work properly with models that are loaded / unloaded several
# times in a pipe (eg VAE).
#
# Requirements:
# - GPU: RTX 3090/ RTX 4090 (24 GB of VRAM)
# - RAM: minimum 48 GB, recommended 64 GB 
#
# It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
# 1) First make sure that the pipeline explictly loads the models in the CPU device 
#   for instance: pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
# 2) Once every potential Lora has been loaded and merged, add the following lines:
#   from mmgp import offload
#   offload.all(pipe)
# The 'transformer' model that contains usually the video or image generator is quantized on the fly by default to 8 bits so that it can fit into 24 GB of VRAM. 
# If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. In that case you need to set the option quantizeTransformer to False to turn off on the fly quantization.
# You can specify a list of additional models string ids to quantize (for instance the text_encoder) using the optional argument modelsToQuantize. This may be useful if you have less than 48 GB of RAM.
# Note that there is little advantage on the GPU / VRAM side to quantize text encoders as their inputs are usually quite light. 
# Conversely if you have more than 64GB RAM you may want to enable RAM pinning with the option pinInRAM = True. You will get in return super fast loading / unloading of models
# (this can save significant time if the same pipeline is run multiple times in a row)
# 
# Sometime there isn't an explicit pipe object as each submodel is loaded separately in the main app. If this is the case, you need to create a dictionary that manually maps all the models.
#
# For instance :
# for flux derived models: pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
# for mochi: pipe = { "text_encoder": self.text_encoder, "transformer": self.dit, "vae":self.decoder }
#
# Please note that there should be always one model whose Id is 'transformer'. It corresponds to the main image / video model which usually needs to be quantized (this is done on the fly by default when loading the model)
# 
# Becareful, lots of models use the T5 XXL as a text encoder. However, quite often their corresponding pipeline configurations point at the official Google T5 XXL repository 
# where there is a huge 40GB model to download and load. It is cumbersorme as it is a 32 bits model and contains the decoder part of T5 that is not used. 
# I suggest you use instead one of the 16 bits encoder only version available around, for instance:
# text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=torch.float16)
#
# Sometime just providing the pipe won't be sufficient as you will need to change the content of the core model: 
# - For instance you may need to disable an existing CPU offload logic that already exists (such as manual calls to move tensors between cuda and the cpu)
# - mmpg to tries to fake the device as being "cuda" but sometimes some code won't be fooled and it will create tensors in the cpu device and this may cause some issues.
# 
# You are free to use my module for non commercial use as long you give me proper credits. You may contact me on twitter @deepbeepmeep
#
# Thanks to
# ---------
# Huggingface / accelerate for the hooking examples
# Huggingface / quanto for their very useful quantizer
# gau-nernst for his Pinnig RAM samples


#
import torch
#
import gc
import time
import functools
from optimum.quanto import freeze, qfloat8, qint8, quantize, QModuleMixin, QTensor



cotenants_map = { 
                             "text_encoder": ["vae", "text_encoder_2"],
                             "text_encoder_2": ["vae", "text_encoder"],                             
                             }

# useful functions to move a group of tensors (to design custom offload patches)
def move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        _dict = {}
        for k, v in obj.items():
            _dict[k] = move_tensors(v, device)
        return _dict
    elif isinstance(obj, list):
        _list = []
        for v in obj:
            _list.append(move_tensors(v, device))
        return _list
    else:
        raise TypeError("Tensor or list / dict of tensors expected")


def get_model_name(model):
    return model.name

class HfHook:
    def __init__(self):
        self.execution_device = "cuda"

    def detach_hook(self, module):
        pass

class offload:
    def __init__(self):
        self.active_models = []
        self.active_models_ids = []
        self.models = {}
        self.verbose = False
        self.models_to_quantize = []
        self.pinned_modules_data = {}
        self.params_of_modules = {}
        self.pinTensors = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check =0

    def collect_module_parameters(self, module: torch.nn.Module, module_params):
        if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            for i in range(len(module)):
                current_layer = module[i]
                module_params.extend(current_layer.parameters())
                module_params.extend(current_layer.buffers())
        else:
            for p in module.parameters(recurse=False):
                module_params.append(p)
            for p in module.buffers(recurse=False):
                module_params.append(p)
            for sub_module in module.children():
                self.collect_module_parameters(sub_module, module_params)

    def can_model_be_cotenant(self, model_id):
        potential_cotenants= cotenants_map.get(model_id, None)
        if potential_cotenants is None: 
            return False
        for existing_cotenant in self.active_models_ids:
            if existing_cotenant not in potential_cotenants: 
                return False    
        return True

    def gpu_load(self, model_id):
        model = self.models[model_id]
        self.active_models.append(model)
        self.active_models_ids.append(model_id)
        if self.verbose:
            model_name = model._get_name()
            print(f"Loading model {model_name} ({model_id}) in GPU")
        if not self.pinInRAM: 
            model.to("cuda")
        else:
            module_params = self.params_of_modules[model_id]            
            for p in module_params:
                if isinstance(p, QTensor):
                    p._data = p._data.cuda(non_blocking=True)             
                    p._scale = p._scale.cuda(non_blocking=True)
                else:
                    p.data = p.data.cuda(non_blocking=True) #
        # torch.cuda.current_stream().synchronize()    
    @torch.compiler.disable()  
    def unload_all(self):
        for model, model_id in zip(self.active_models, self.active_models_ids):
            if not self.pinInRAM:
                model.to("cpu")
            else:
                module_params = self.params_of_modules[model_id]            
                pinned_parameters_data = self.pinned_modules_data[model_id]
                for p in module_params:
                    if isinstance(p, QTensor):
                        data = pinned_parameters_data[p]
                        p._data = data[0]          
                        p._scale = data[1]             
                    else:
                        p.data = pinned_parameters_data[p]            
               
 
        self.active_models = []
        self.active_models_ids = []
        torch.cuda.empty_cache()
        gc.collect()

    def move_args_to_gpu(self, *args, **kwargs):
        new_args= []
        new_kwargs={}
        for arg in args:
            if torch.is_tensor(arg):    
                if arg.dtype == torch.float32:
                    arg = arg.to(torch.bfloat16).cuda(non_blocking=True)             
                else:
                    arg = arg.cuda(non_blocking=True)             
            new_args.append(arg)

        for k in kwargs:
            arg = kwargs[k]
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(torch.bfloat16).cuda(non_blocking=True)             
                else:
                    arg = arg.cuda(non_blocking=True)             
            new_kwargs[k]= arg
        
        return new_args, new_kwargs

    def ready_to_check_mem(self, forceMemoryCheck):
        cur_clock = time.time()
        # can't check at each call if we can empty the cuda cache as quering the reserved memory value is a time consuming operation
        if not forceMemoryCheck and (cur_clock - self.last_reserved_mem_check)<0.200:
            return False
        self.last_reserved_mem_check = cur_clock
        return True        


    def empty_cache_if_needed(self):
        mem_reserved = torch.cuda.memory_reserved()
        if mem_reserved >= 0.9*self.device_mem_capacity:            
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved: 
                # print(f"Cuda empty cache triggered as Allocated Memory ({mem_allocated/1024000:0f} MB) is lot less than Cached Memory ({mem_reserved/1024000:0f} MB)  ")
                torch.cuda.empty_cache()
                # print(f"New cached memory after purge is {torch.cuda.memory_reserved()/1024000:0f} MB)  ")

    def hook_me_light(self, target_module, forceMemoryCheck, previous_method):
        def check_empty_cache(module, *args, **kwargs):
            if self.ready_to_check_mem(forceMemoryCheck):
                self.empty_cache_if_needed()
            return previous_method(*args, **kwargs) 
  
        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_empty_cache, target_module), previous_method) )

        
    def hook_me(self, target_module, model, model_id, module_id, previous_method):
        def check_change_module(module, *args, **kwargs):
            performEmptyCacheTest = False
            if not model_id in self.active_models_ids:
                new_model_id = getattr(module, "_mm_id") 
                # do not always unload existing models if it is more efficient to keep in them in the GPU 
                # (e.g: small modules whose calls are text encoders) 
                if not self.can_model_be_cotenant(new_model_id) :
                    self.unload_all()
                    performEmptyCacheTest = False
                self.gpu_load(new_model_id)
            # transfer leftovers inputs that were incorrectly created in the RAM (mostly due to some .device tests that returned incorrectly "cpu")
            args, kwargs = self.move_args_to_gpu(*args, **kwargs)
            if performEmptyCacheTest:
                self.empty_cache_if_needed()
            return previous_method(*args, **kwargs) 
  
        if hasattr(target_module, "_mm_id"):
            return
        setattr(target_module, "_mm_id", model_id)

        # create a fake accelerate parameter so that the _execution_device property returns always "cuda" 
        # (it is queried in many pipelines even if offloading is not properly implemented)  
        if not hasattr(target_module, "_hf_hook"):
            setattr(target_module, "_hf_hook", HfHook())
        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method) )

        if not self.verbose:
            return

        if module_id == None or module_id =='':
            model_name = model._get_name()
            print(f"Hooked in model '{model_id}' ({model_name})")


    # Not implemented yet, but why would one want to get rid of these features ?
    # def unhook_module(module: torch.nn.Module):
    #     if not hasattr(module,"_mm_id"):
    #         return
        
    #     delattr(module, "_mm_id")
                 
    # def unhook_all(parent_module: torch.nn.Module):
    #     for module in parent_module.components.items():
    #         self.unhook_module(module)


  

    @classmethod
    def all(cls, pipe_or_dict_of_modules, quantizeTransformer = True, pinInRAM = False,  verbose = True, modelsToQuantize = None ):
        self = cls()
        self.verbose = verbose
        self.pinned_modules_data = {}

        # compile not working yet or slower
        compile = False
        self.pinInRAM = pinInRAM 
        pipe = None
        preloadInRAM = True
        torch.set_default_device('cuda')
        if hasattr(pipe_or_dict_of_modules, "components"):
            pipe_or_dict_of_modules.to("cpu") #XXXX
            # create a fake Accelerate parameter so that lora loading doesn't change the device
            pipe_or_dict_of_modules.hf_device_map = torch.device("cuda")
            pipe = pipe_or_dict_of_modules
            pipe_or_dict_of_modules= pipe_or_dict_of_modules.components 
        
        
        models = {k: v for k, v in pipe_or_dict_of_modules.items() if isinstance(v, torch.nn.Module)}

        modelsToQuantize =  modelsToQuantize if modelsToQuantize is not None else []
        if not isinstance(modelsToQuantize, list):
            modelsToQuantize = [modelsToQuantize]
        if quantizeTransformer:
            modelsToQuantize.append("transformer")
        self.models_to_quantize = modelsToQuantize
 #       del  models["transformer"] # to test everything but the transformer that has a much longer loading
 #       models = { 'transformer': pipe_or_dict_of_modules["transformer"]} # to test only the transformer
        for model_id in models: 
            current_model: torch.nn.Module = models[model_id] 
            # make sure that no RAM or GPU memory is not allocated for gradiant / training
            current_model.to("cpu").eval() #XXXXX

            # Quantize model just before transferring it to the RAM to keep OS cache file
            # open as short as possible. Indeed it seems that as long as the lazy safetensors 
            # are not fully fully loaded, the OS won't be able to release the corresponding cache file in RAM.
            if model_id in self.models_to_quantize:
                print(f"Quantization of model '{model_id}' started")
                quantize(current_model, weights=qint8)
                freeze(current_model)
                print(f"Quantization of model '{model_id}' done")
                torch.cuda.empty_cache()
                gc.collect()         


    
            if preloadInRAM: # 
                # load all the remaining unread lazy safetensors in RAM to free open cache files 
                for p in current_model.parameters():
                    # Preread every tensor in RAM except tensors that have just been quantified
                    # and are no longer needed
                    if isinstance(p, QTensor):
                        # fix quanto bug (see below) now as he won't have any opportunity to do it during RAM pinning 
                        if not pinInRAM and p._scale.dtype == torch.float32:
                            p._scale = p._scale.to(torch.bfloat16) 

                    else:
                        if p.data.dtype == torch.float32:
                            # convert any left overs float32 weight to bloat16 to divide by 2 the model memory footprint
                            p.data = p.data.to(torch.bfloat16)
                        else:
                            # force reading the tensors from the disk by pretending to modify them
                            p.data = p.data + 0
                     

            addModelFlag  = False

            current_block_sequence = None
            for submodule_name, submodule in current_model.named_modules():  
                if hasattr(submodule, "forward"):
                    submodule_method = getattr(submodule, "forward")
                    if callable(submodule_method):   
                        addModelFlag  = True 
                        if submodule_name=='' or len(submodule_name.split("."))==1:
                            # hook only the first two levels of  modules with the full suite of processing                             
                            self.hook_me(submodule, current_model, model_id, submodule_name, submodule_method)
                        else:
                            forceMemoryCheck = False
                            pos = submodule_name.find(".0.") 
                            if pos > 0:
                                if current_block_sequence ==  None:
                                    new_candidate = submodule_name[0:pos+3]
                                    if len(new_candidate.split("."))<=4:
                                        current_block_sequence = new_candidate  
                                        # force a memory check when initiating a new sequence of blocks as the shapes of tensor will certainly change
                                        # and memory reusability is less likely
                                        # we limit this check to the first level of blocks as quering the cuda cache is time consuming
                                        forceMemoryCheck = True
                                else:
                                    if current_block_sequence != submodule_name[0:len(current_block_sequence)]:
                                         current_block_sequence = None
                            self.hook_me_light(submodule, forceMemoryCheck, submodule_method)

                            
            if addModelFlag:
                if model_id not in self.models:
                    self.models[model_id] = current_model

        # Pin in RAM models only once they have been fully loaded otherwise there may be some contention in the non pageable memory
        # between partially loaded lazy safetensors and pinned tensors   
        if pinInRAM: 
            if verbose:
                print("Pinning model tensors in RAM")
            torch.cuda.empty_cache()
            gc.collect()                     
            for model_id in models: 
                pinned_parameters_data = {}
                current_model: torch.nn.Module = models[model_id] 
                for p in current_model.parameters():
                    if isinstance(p, QTensor):
                        # pin in memory both quantized data and scales of quantized parameters
                        # but don't pin .data as it corresponds to the original tensor that we don't want to reload
                        p._data = p._data.pin_memory()
                        # fix quanto bug that allows _scale to be float32 if the original weight was float32 
                        # (this may cause type mismatch between dequantified bfloat16 weights and float32 scales)
                        p._scale = p._scale.to(torch.bfloat16).pin_memory() if p._scale.dtype == torch.float32 else p._scale.pin_memory()
                        pinned_parameters_data[p]=[p._data, p._scale]
                    else:
                        p.data = p.data.pin_memory() 
                        pinned_parameters_data[p]=p.data 
                for b in current_model.buffers():
                    b.data = b.data.pin_memory()

                pinned_buffers_data = {b: b.data for b in current_model.buffers()}
                pinned_parameters_data.update(pinned_buffers_data)
                self.pinned_modules_data[model_id]=pinned_parameters_data

                module_params = []      
                self.params_of_modules[model_id] = module_params           
                self.collect_module_parameters(current_model,module_params)

        if compile:
            if verbose:
                print("Torch compilation started")
            torch._dynamo.config.cache_size_limit = 10000
            # if pipe != None and hasattr(pipe, "__call__"):
            #     pipe.__call__= torch.compile(pipe.__call__, mode= "max-autotune")

            for model_id in models: 
                    current_model: torch.nn.Module = models[model_id]
                    current_model.compile(mode= "max-autotune")                                 
            #models["transformer"].compile()
                
            if verbose:
                print("Torch compilation done")

        torch.cuda.empty_cache()
        gc.collect()         


        return self

            
