# ------------------ Memory Management 2.0 for the GPU Poor by DeepBeepMeep (mmgp)------------------
#
# This module contains multiples optimisations so that models such as Flux (and derived), Mochi, CogView, HunyuanVideo, ...  can run smoothly on a 24 GB GPU limited card. 
# This a replacement for the accelerate library that should in theory manage offloading, but doesn't work properly with models that are loaded / unloaded several
# times in a pipe (eg VAE).
#
# Requirements:
# - VRAM: minimum 12 GB, recommended 24 GB (RTX 3090/ RTX 4090)
# - RAM: minimum 24 GB, recommended 48 - 64 GB 
#
# It is almost plug and play and just needs to be invoked from the main app just after the model pipeline has been created.
# 1) First make sure that the pipeline explictly loads the models in the CPU device 
#   for instance: pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
# 2) Once every potential Lora has been loaded and merged, add the following lines:
# For a quick setup, you may want to choose between 4 profiles depending on your hardware, for instance:
#   from mmgp import offload, profile_type
#   offload.profile(pipe, profile_type.HighRAM_LowVRAM_Fast)
# Alternatively you may want to your own parameters, for instance:
#   from mmgp import offload
#   offload.all(pipe, pinInRAM=true, modelsToQuantize = ["text_encoder_2"] )
# The 'transformer' model that contains usually the video or image generator is quantized on the fly by default to 8 bits so that it can fit into 24 GB of VRAM. 
# If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. In that case you need to set the option quantizeTransformer to False to turn off on the fly quantization.
# You can specify a list of additional models string ids to quantize (for instance the text_encoder) using the optional argument modelsToQuantize. This may be useful if you have less than 48 GB of RAM.
# Note that there is little advantage on the GPU / VRAM side to quantize text encoders as their inputs are usually quite light. 
# Conversely if you have more than 48GB RAM you may want to enable RAM pinning with the option pinInRAM = True. You will get in return super fast loading / unloading of models
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
import sys
import json

from optimum.quanto import freeze, qfloat8, qint8, quantize, QModuleMixin, QTensor



ONE_MB =  1048576

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

def _quantize(model_to_quantize, weights=qint8, verboseLevel = 1, threshold = 1000000000, model_id = None):
    
    sizeofbfloat16 = torch.bfloat16.itemsize

    def compute_submodule_size(submodule):
        size = 0
        for p in submodule.parameters(recurse=False):
            size  += torch.numel(p.data) * sizeofbfloat16

        for p in submodule.buffers(recurse=False):
            size  += torch.numel(p.data) * sizeofbfloat16

        return size
    
    total_size =0
    total_excluded = 0
    exclude_list = []
    submodule_size = 0
    submodule_names = []
    cur_blocks_prefix = None
    prev_blocks_prefix = None

    print(f"Quantization of model '{model_id}' started")

    for submodule_name, submodule in model_to_quantize.named_modules():  
        if isinstance(submodule, QModuleMixin):
            if verboseLevel>=1:
                print("No quantization to do as model is already quantized")
            return False


        if submodule_name=='':
            continue


        flush = False
        if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            if cur_blocks_prefix == None:
                cur_blocks_prefix = submodule_name + "."
                flush = True                    
            else:
                #if cur_blocks_prefix != submodule_name[:len(cur_blocks_prefix)]:
                if not submodule_name.startswith(cur_blocks_prefix):
                    cur_blocks_prefix = submodule_name + "."
                    flush = True                    
        else:                
            if cur_blocks_prefix is not None:
                #if not cur_blocks_prefix == submodule_name[0:len(cur_blocks_prefix)]:
                if not submodule_name.startswith(cur_blocks_prefix):
                    cur_blocks_prefix = None 
                    flush = True                    

        if flush:
            if submodule_size <= threshold:
                exclude_list += submodule_names
                if verboseLevel >=2:
                    print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
                total_excluded += submodule_size

            submodule_size = 0
            submodule_names = []
        prev_blocks_prefix = cur_blocks_prefix
        size = compute_submodule_size(submodule)
        submodule_size += size
        total_size += size
        submodule_names.append(submodule_name)

    if submodule_size > 0 and submodule_size <= threshold:
        exclude_list += submodule_names
        if verboseLevel >=2:
            print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
        total_excluded += submodule_size

    perc_excluded =total_excluded/ total_size if total_size >0 else 1
    if verboseLevel >=2:
        print(f"Total Excluded {total_excluded/ONE_MB:.1f} MB oF {total_size/ONE_MB:.1f} that is {perc_excluded*100:.2f}%")
    if perc_excluded >= 0.10:
        print(f"Too many many modules are excluded, there is something wrong with the selection, switch back to full quantization.")
        exclude_list = None

            # we are obviously loading a model that has been already quantized

    quantize(model_to_quantize,weights, exclude= exclude_list)
    freeze(model_to_quantize)
    torch.cuda.empty_cache()
    gc.collect()         
    print(f"Quantization of model '{model_id}' done")

    return True

def get_model_name(model):
    return model.name

import enum    
class profile_type(int, enum.Enum): 
    HighRAM_HighVRAM_Fastest = 1
    HighRAM_LowVRAM_Fast = 2
    LowRAM_HighVRAM_Medium = 3
    LowRAM_LowVRAM_Slow = 4
    VerylowRAM_LowVRAM_Slowest = 5

class HfHook:
    def __init__(self):
        self.execution_device = "cuda"

    def detach_hook(self, module):
        pass

class offload:
    def __init__(self):
        self.active_models = []
        self.active_models_ids = []
        self.active_subcaches = {}        
        self.models = {}
        self.verboseLevel = 0
        self.models_to_quantize = []
        self.pinned_modules_data = {}
        self.blocks_of_modules = {}
        self.blocks_of_modules_sizes = {}
        self.compile = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check =0
        self.loaded_blocks = {}
        self.prev_blocks_names = {}
        self.next_blocks_names = {}
        self.default_stream = torch.cuda.default_stream(torch.device("cuda")) # torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        self.async_transfers = False


    def add_module_to_blocks(self, model_id, blocks_name, submodule, prev_block_name):

        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if entry_name in self.blocks_of_modules:
            blocks_params = self.blocks_of_modules[entry_name]
            blocks_params_size = self.blocks_of_modules_sizes[entry_name]
        else:
            blocks_params = []
            self.blocks_of_modules[entry_name] = blocks_params
            blocks_params_size = 0
            if blocks_name !=None:
                prev_entry_name = None if prev_block_name == None else  model_id + "/" + prev_block_name
                self.prev_blocks_names[entry_name] =  prev_entry_name
                if not prev_block_name == None:
                    self.next_blocks_names[prev_entry_name] = entry_name        

        for p in submodule.parameters(recurse=False):
            blocks_params.append(p)
            if isinstance(p, QTensor):
                blocks_params_size += p._data.nbytes
                blocks_params_size += p._scale.nbytes
            else:
                blocks_params_size += p.data.nbytes

        for p in submodule.buffers(recurse=False):
            blocks_params.append(p)      
            blocks_params_size += p.data.nbytes


        self.blocks_of_modules_sizes[entry_name] = blocks_params_size

        return blocks_params_size


    def can_model_be_cotenant(self, model_id):
        potential_cotenants= cotenants_map.get(model_id, None)
        if potential_cotenants is None: 
            return False
        for existing_cotenant in self.active_models_ids:
            if existing_cotenant not in potential_cotenants: 
                return False    
        return True

    @torch.compiler.disable()    
    def gpu_load_blocks(self, model_id, blocks_name, async_load = False):
        if blocks_name != None:
            self.loaded_blocks[model_id] = blocks_name           

        def cpu_to_gpu(stream_to_use, blocks_params, record_for_stream = None):
            with torch.cuda.stream(stream_to_use):
                for p in blocks_params:
                    if isinstance(p, QTensor):
                        p._data = p._data.cuda(non_blocking=True)             
                        p._scale = p._scale.cuda(non_blocking=True)
                    else:
                        p.data = p.data.cuda(non_blocking=True) 
                    
                    if record_for_stream != None:
                        if isinstance(p, QTensor):
                            p._data.record_stream(record_for_stream)
                            p._scale.record_stream(record_for_stream)
                        else:
                            p.data.record_stream(record_for_stream)


        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            print(f"Loading model {entry_name} ({model_name}) in GPU")
 

        if self.async_transfers and blocks_name != None:
            first = self.prev_blocks_names[entry_name] == None
            next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
            if first:
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules[entry_name])
                # if next_blocks_entry != None:
                #     self.transfer_stream.wait_stream(self.default_stream)
            # else:
            #     self.transfer_stream.wait_stream(self.default_stream)
            torch.cuda.synchronize()

            if next_blocks_entry != None:
                cpu_to_gpu(self.transfer_stream, self.blocks_of_modules[next_blocks_entry]) #, self.default_stream

        else:
            # if self.async_transfers:
            #     self.transfer_stream.wait_stream(self.default_stream)
            cpu_to_gpu(self.default_stream, self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()
 

    @torch.compiler.disable()  
    def gpu_unload_blocks(self, model_id, blocks_name):
        if blocks_name != None:
            self.loaded_blocks[model_id] = None 

        blocks_name = model_id if blocks_name is None else model_id + "/" + blocks_name

        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            print(f"Unloading model {blocks_name} ({model_name}) from GPU")
 
        blocks_params = self.blocks_of_modules[blocks_name]

        if model_id in self.pinned_modules_data:
            pinned_parameters_data = self.pinned_modules_data[model_id]
            for p in blocks_params:
                if isinstance(p, QTensor):
                    data = pinned_parameters_data[p]
                    p._data = data[0]          
                    p._scale = data[1]             
                else:
                    p.data = pinned_parameters_data[p]
        else:
            for p in blocks_params:
                if isinstance(p, QTensor):
                    p._data = p._data.cpu()
                    p._scale = p._scale.cpu()     
                else:
                    p.data = p.data.cpu()           



    @torch.compiler.disable()  
    def gpu_load(self, model_id):
        model = self.models[model_id]
        self.active_models.append(model)
        self.active_models_ids.append(model_id)

        self.gpu_load_blocks(model_id, None)

        # torch.cuda.current_stream().synchronize()    

    def unload_all(self):
        for model_id in self.active_models_ids:
            self.gpu_unload_blocks(model_id, None)      
            loaded_block = self.loaded_blocks[model_id]
            if loaded_block != None:
                self.gpu_unload_blocks(model_id, loaded_block)      
                self.loaded_blocks[model_id] = None  
 
        self.active_models = []
        self.active_models_ids = []
        self.active_subcaches = []
        torch.cuda.empty_cache()
        gc.collect()
        self.last_reserved_mem_check = time.time()

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

    def ready_to_check_mem(self):
        if self.compile:
            return
        cur_clock = time.time()
        # can't check at each call if we can empty the cuda cache as quering the reserved memory value is a time consuming operation
        if (cur_clock - self.last_reserved_mem_check)<0.200:
            return False
        self.last_reserved_mem_check = cur_clock
        return True        


    def empty_cache_if_needed(self):
        mem_reserved = torch.cuda.memory_reserved()
        mem_threshold = 0.9*self.device_mem_capacity
        if mem_reserved >= mem_threshold:            
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved: 
                # print(f"Cuda empty cache triggered as Allocated Memory ({mem_allocated/1024000:0f} MB) is lot less than Cached Memory ({mem_reserved/1024000:0f} MB)  ")
                torch.cuda.empty_cache()
                tm= time.time()
                if self.verboseLevel >=2:
                    print(f"Empty Cuda cache at {tm}")
                # print(f"New cached memory after purge is {torch.cuda.memory_reserved()/1024000:0f} MB)  ")


    def any_param_or_buffer(self, target_module: torch.nn.Module):
        
        for _ in target_module.parameters(recurse= False):
            return True
        
        for _ in target_module.buffers(recurse= False):
            return True
        
        return False



    def hook_me_light(self, target_module, model_id,blocks_name, previous_method, context):

        anyParam = self.any_param_or_buffer(target_module)

        def check_empty_cuda_cache(module, *args, **kwargs):
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()
            return previous_method(*args, **kwargs) 


        def load_module_blocks(module,  *args, **kwargs):
            #some_context = context #for debugging
            if blocks_name == None:
                if self.ready_to_check_mem():
                    self.empty_cache_if_needed()
            else:                
                loaded_block = self.loaded_blocks[model_id]
                if (loaded_block == None or loaded_block != blocks_name) :
                    if loaded_block != None:
                        self.gpu_unload_blocks(model_id, loaded_block)
                        if self.ready_to_check_mem():
                            self.empty_cache_if_needed()
                    self.loaded_blocks[model_id] = blocks_name
                    self.gpu_load_blocks(model_id, blocks_name)
            return previous_method(*args, **kwargs) 

        if hasattr(target_module, "_mm_id"):
            orig_model_id = getattr(target_module, "_mm_id")
            if self.verboseLevel >=2:
                print(f"Model '{model_id}' shares module '{target_module._get_name()}' with module '{orig_model_id}' ")
            assert not anyParam
            return
        setattr(target_module, "_mm_id", model_id)


        if  blocks_name != None and anyParam: 
            setattr(target_module, "forward", functools.update_wrapper(functools.partial(load_module_blocks, target_module), previous_method) )
            #print(f"new cache:{blocks_name}")
        else:
            setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_empty_cuda_cache, target_module), previous_method) )

        
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

        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method) )

        if not self.verboseLevel >=1:
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


    @staticmethod
    def fast_load_transformers_model(model_path: str):
        """
        quick version of .LoadfromPretrained of  the transformers library
        used to build a model and load the corresponding weights (quantized or not)
        """       

        from transformers import AutoConfig

        if model_path.endswith(".sft") or model_path.endswith(".safetensors"):
            config_path = model_path[ : model_path.rfind("/")]
        else:
            raise("full model path expected")
        config_fullpath = config_path +"/config.json"

        import os.path
        if not os.path.isfile(config_fullpath):
            raise("a 'config.json' that describes the model is required in the directory of the model")

        with open(config_fullpath, "r", encoding="utf-8") as reader:
            text = reader.read()
        transformer_config= json.loads(text)
        architectures = transformer_config["architectures"]
        class_name = architectures[0] 

        module = __import__("transformers")
        transfomer_class = getattr(module, class_name)
 
        config = AutoConfig.from_pretrained(config_path)     

        from accelerate import init_empty_weights
        #needed to keep inits of non persistent buffers
        with init_empty_weights():
            model = transfomer_class(config)
        
        model = model.base_model
        torch.set_default_device('cpu')
        model.apply(model._initialize_weights)

        #missing_keys, unexpected_keys =  
        offload.load_model_data(model,model_path, strict = True )

        return model
        # #            text_encoder.final_layer_norm = text_encoder.norm
    #     model = model.base_model
    #     model.final_layer_norm = model.norm
    #     self.model = model



    @staticmethod
    def load_model_data(model, file_path: str, device=torch.device('cpu'), strict = True):
        """
        Load a model, detect if it has been previously quantized using quanto and do the extra setup if necessary
        """
        from optimum.quanto import requantize
        import safetensors.torch 

        if "quanto" in file_path.lower():
            pos = str.rfind(file_path, ".")
            if pos > 0:
                quantization_map_path = file_path[:pos]
            quantization_map_path += "_map.json"


            with open(quantization_map_path, 'r') as f:
                quantization_map = json.load(f)

            state_dict = safetensors.torch.load_file(file_path)

            # change dtype of current meta model parameters because 'requantize' won't update the dtype on non quantized parameters
            for k, p in model.named_parameters():
                if not k in quantization_map and k in state_dict:
                    p_in_sd = state_dict[k] 
                    if p.data.dtype != p_in_sd.data.dtype:
                         p.data = p.data.to(p_in_sd.data.dtype)

            requantize(model, state_dict, quantization_map, device)    

            # for k, p in model.named_parameters():
            #     if p.data.dtype == torch.float32:
            #         pass        
            

            # del state_dict
            return

        else:
            if ".safetensors" in file_path or ".sft" in file_path: 
                state_dict = safetensors.torch.load_file(file_path)

            else:

                state_dict = torch.load(file_path, weights_only=True)
                if "module" in state_dict:
                    state_dict = state_dict["module"]


        model.load_state_dict(state_dict, strict = strict,  assign = True ) #strict=True,


        return

    @staticmethod
    def save_model(model, file_path, do_quantize = False, quantization_type = qint8 ):
        """save the weights of a model and quantize them if requested
        These weights can be loaded again using 'load_model_data'
        """       
        import safetensors.torch
        pos = str.rfind(file_path, ".")
        if pos > 0:
            file_path = file_path[:pos]
        
        if do_quantize:
            _quantize(model, weights=quantization_type)
        
# #        state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
#         state_dict = {k: v  for k, v in model.state_dict().items()}



        safetensors.torch.save_file(model.state_dict(), file_path + '.safetensors')

        if do_quantize:
            from optimum.quanto import quantization_map

            with open(file_path + '_map.json', 'w') as f:
                json.dump(quantization_map(model), f)

  

    @classmethod
    def all(cls, pipe_or_dict_of_modules, quantizeTransformer = True, pinInRAM = False,  verboseLevel = 1, modelsToQuantize = None, budgets= 0, info = None):
        """Hook to a pipeline or a group of modules in order to reduce their VRAM requirements:
        pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
        quantizeTransformer: set True by default will quantize on the fly the video / image model
        pinInRAM: move models in reserved memor. This allows very fast performance but requires 50% extra RAM (usually >=64 GB)
        modelsToQuantize: a list of models to be also quantized on the fly (e.g the text_encoder), useful to reduce bith RAM and VRAM consumption
        budgets: 0 by default (unlimited). If non 0, it corresponds to the maximum size in MB that every model will occupy at any moment
         (in fact the real usage is twice this number). It is very efficient to reduce VRAM consumption but this feature may be very slow
         if pinInRAM is not enabled
        """        
        
        self = cls()
        self.verboseLevel = verboseLevel
        self.pinned_modules_data = {}
        model_budgets = {}

#        model_budgets = {"text_encoder_2": 3400 }
        HEADER = '\033[95m'
        ENDC = '\033[0m'
        BOLD ='\033[1m'
        UNBOLD ='\033[0m'
        
        print(f"{BOLD}{HEADER}************ Memory Management for the GPU Poor (mmgp 2.0) by DeepBeepMeep ************{ENDC}{UNBOLD}")
        if info != None:
            print(info)
        budget = 0
        if not budgets is None:
            if isinstance(budgets , dict):
                model_budgets = budgets
            else:
                budget = int(budgets) * ONE_MB

        if (budgets!= None or budget >0) :
            self.async_transfers = True

        #pinInRAM = True
        # compile not working yet or slower
        compile = False # True
        #quantizeTransformer = False
        #self.async_transfers = False
        self.compile = compile

        pipe = None
        torch.set_default_device('cuda')
        if hasattr(pipe_or_dict_of_modules, "components"):
            # commented as it not very useful and generates warnings
            #pipe_or_dict_of_modules.to("cpu") #XXXX
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
        models_already_loaded = []

        modelsToPin = None
        pinAllModels = False
        if isinstance(pinInRAM, bool):
            pinAllModels = pinInRAM
        elif isinstance(pinInRAM, list):            
            modelsToPin = pinInRAM
        else:
            modelsToPin = [pinInRAM]

 #       del  models["transformer"] # to test everything but the transformer that has a much longer loading
        sizeofbfloat16 = torch.bfloat16.itemsize
   #        
   #     models = { 'transformer': pipe_or_dict_of_modules["transformer"]} # to test only the transformer


        for model_id in models: 
            current_model: torch.nn.Module = models[model_id] 
            modelPinned = pinAllModels or (modelsToPin != None and model_id in modelsToPin)
            # make sure that no RAM or GPU memory is not allocated for gradiant / training
            current_model.to("cpu").eval()
            already_loaded = False
            # Quantize model just before transferring it to the RAM to keep OS cache file
            # open as short as possible. Indeed it seems that as long as the lazy safetensors 
            # are not fully fully loaded, the OS won't be able to release the corresponding cache file in RAM.
            if model_id in self.models_to_quantize:

                already_quantized = _quantize(current_model, weights=qint8, verboseLevel = self.verboseLevel, model_id=model_id)
                if not already_quantized:
                    already_loaded = True
                    models_already_loaded.append(model_id) 


            current_model_size = 0    
            # load all the remaining unread lazy safetensors in RAM to free open cache files 
            for p in current_model.parameters():
                # Preread every tensor in RAM except tensors that have just been quantified
                # and are no longer needed
                if isinstance(p, QTensor):
                    # fix quanto bug (see below) now as he won't have any opportunity to do it during RAM pinning 
                    if not modelPinned and p._scale.dtype == torch.float32:
                        p._scale = p._scale.to(torch.bfloat16) 
                    current_model_size +=  torch.numel(p._scale) * sizeofbfloat16
                    current_model_size +=  torch.numel(p._data) * sizeofbfloat16 / 2
                    if pinInRAM and not already_loaded:
                        # Force flushing the lazy load so that reserved memory can be freed when we are ready to pin
                        p._scale = p._scale + 0
                        p._data = p._data + 0
                else:
                    if p.data.dtype == torch.float32:
                        # convert any left overs float32 weight to bloat16 to divide by 2 the model memory footprint
                        p.data = p.data.to(torch.bfloat16)
                    else:
                        # force reading the tensors from the disk by pretending to modify them
                        p.data = p.data + 0

                    current_model_size +=  torch.numel(p.data) * p.data.element_size()
                             
            for b in current_model.buffers():
                if b.data.dtype == torch.float32:
                    # convert any left overs float32 weight to bloat16 to divide by 2 the model memory footprint
                    b.data = b.data.to(torch.bfloat16)
                else:
                    # force reading the tensors from the disk by pretending to modify them
                    b.data = b.data + 0

                current_model_size +=  torch.numel(p.data) * p.data.element_size()

            if model_id not in self.models:
                self.models[model_id] = current_model

            
            model_budget = model_budgets[model_id] * ONE_MB if model_id in model_budgets else budget

            if  model_budget > 0 and model_budget > current_model_size:
                model_budget = 0
            
            model_budgets[model_id] = model_budget

        # Pin in RAM models only once they have been fully loaded otherwise there will be some contention (at least on Linux OS) in the non pageable memory
        # between partially loaded lazy safetensors and pinned tensors   
        for model_id in models: 
            current_model: torch.nn.Module = models[model_id] 
            if not (pinAllModels or modelsToPin != None and model_id in modelsToPin):
                continue
            if verboseLevel>=1:
                print(f"Pinning tensors of '{model_id}' in RAM")
            gc.collect()                     
            pinned_parameters_data = {}
            for p in current_model.parameters():
                if isinstance(p, QTensor):
                    # pin in memory both quantized data and scales of quantized parameters
                    # but don't pin .data as it corresponds to the original tensor that we don't want to reload
                    p._data = p._data.pin_memory()
                    # fix quanto bug (that seems to have been fixed since&) that allows _scale to be float32 if the original weight was float32 
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


        #  Hook forward methods of modules 
        for model_id in models: 
            current_model: torch.nn.Module = models[model_id] 
            current_budget = model_budgets[model_id]
            current_size = 0
            cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq = None, None, None, -1
            self.loaded_blocks[model_id] = None
             
            for submodule_name, submodule in current_model.named_modules():  
                # create a fake accelerate parameter so that the _execution_device property returns always "cuda" 
                # (it is queried in many pipelines even if offloading is not properly implemented)  
                if  not hasattr(submodule, "_hf_hook"):
                    setattr(submodule, "_hf_hook", HfHook())

                if submodule_name=='':
                    continue

                if current_budget > 0:
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
                        if cur_blocks_prefix == None:
                            cur_blocks_prefix = submodule_name + "."
                        else:
                            #if cur_blocks_prefix != submodule_name[:len(cur_blocks_prefix)]:
                            if not submodule_name.startswith(cur_blocks_prefix):
                                cur_blocks_prefix = submodule_name + "."
                                cur_blocks_name,cur_blocks_seq = None, -1
                    else:
                        
                        if cur_blocks_prefix is not None:
                            #if cur_blocks_prefix == submodule_name[0:len(cur_blocks_prefix)]:
                            if submodule_name.startswith(cur_blocks_prefix):
                                num = int(submodule_name[len(cur_blocks_prefix):].split(".")[0])
                                if num != cur_blocks_seq and (cur_blocks_name == None or current_size > current_budget): 
                                    prev_blocks_name = cur_blocks_name
                                    cur_blocks_name = cur_blocks_prefix + str(num)
                                    # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
                                cur_blocks_seq = num
                            else:
                                cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq = None, None, None, -1

                if hasattr(submodule, "forward"):
                    submodule_method = getattr(submodule, "forward")
                    if callable(submodule_method):   
                        if len(submodule_name.split("."))==1:
                            # hook only the first level of modules with the full suite of processing                             
                            self.hook_me(submodule, current_model, model_id, submodule_name, submodule_method)
                        else: 
                            # force a memory check when initiating a new sequence of blocks as the shapes of tensor will certainly change
                            # and memory reusability is less likely
                            # we limit this check to the first level of blocks as quering the cuda cache is time consuming
                            self.hook_me_light(submodule, model_id, cur_blocks_name, submodule_method, context = submodule_name)

                        if compile and cur_blocks_name != None and model_id == "transformer" and "_blocks" in submodule_name:
                             submodule.compile(mode="reduce-overhead" ) #mode= "max-autotune"     

                        current_size = self.add_module_to_blocks(model_id, cur_blocks_name, submodule, prev_blocks_name)


        if compile and False:
            if verboseLevel>=1:
                print("Torch compilation started")
            torch._dynamo.config.cache_size_limit = 10000
            # if pipe != None and hasattr(pipe, "__call__"):
            #     pipe.__call__= torch.compile(pipe.__call__, mode= "max-autotune")

            for model_id in models: 
                    current_model: torch.nn.Module = models[model_id]
                    current_model.compile(mode= "max-autotune")                                 
            #models["transformer"].compile()
                
            if verboseLevel>=1:
                print("Torch compilation done")

        if verboseLevel >=2:
            for n,b in self.blocks_of_modules_sizes.items():
                print(f"Size of submodel '{n}': {b/ONE_MB:.1f} MB")

        torch.cuda.empty_cache()
        gc.collect()         

        return self



    @staticmethod
    def profile(pipe_or_dict_of_modules,profile_no: profile_type, quantizeTransformer = True):
        """Apply a configuration profile that depends on your hardware:
        pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
        profile_name : num of the profile:
            HighRAM_HighVRAM_Fastest (=1): at least 48 GB of RAM and 24 GB of VRAM : the fastest well suited for a RTX 3090 / RTX 4090
            HighRAM_LowVRAM_Fast (=2): at least 48 GB of RAM and 12 GB of VRAM : a bit slower, better suited for RTX 3070/3080/4070/4080 
                or for RTX 3090 / RTX 4090 with large pictures batches or long videos
            LowRAM_HighVRAM_Medium (=3): at least 32 GB of RAM and 24 GB of VRAM : so so speed but adapted for RTX 3090 / RTX 4090 with limited RAM
            LowRAM_LowVRAM_Slow (=4): at least 32 GB of RAM and 12 GB of VRAM : if have little VRAM or generate longer videos 
            VerylowRAM_LowVRAM_Slowest (=5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work
        quantizeTransformer: bool = True, the main model is quantized by default for all the profiles, you may want to disable that to get the best image quality
        """      


        modules = pipe_or_dict_of_modules
        if hasattr(modules, "components"):
            modules= modules.components 
        any_T5 = False
        if "text_encoder_2" in modules:
            text_encoder_2 =  modules["text_encoder_2"]
            any_T5 = "t5" in text_encoder_2.__module__.lower()
        extra_mod_to_quantize = ("text_encoder_2" if any_T5 else "text_encoder")  

        # transformer (video or image generator) should be as small as possible to not occupy space that could be used by actual image data
        # on the other hand the text encoder should be quite large (as long as it fits in 10 GB of VRAM) to reduce sequence offloading

        budgets = { "transformer" : 600 , "text_encoder": 3000, "text_encoder_2": 3000 }

        if profile_no == profile_type.HighRAM_HighVRAM_Fastest:
            info = "You have chosen a Very Fast profile that requires at least 48 GB of RAM and 24 GB of VRAM."
            return offload.all(pipe_or_dict_of_modules, pinInRAM= True, info = info, quantizeTransformer= quantizeTransformer)
        elif profile_no == profile_type.HighRAM_LowVRAM_Fast:
            info = "You have chosen a Fast profile that requires at least 48 GB of RAM and 12 GB of VRAM."
            return offload.all(pipe_or_dict_of_modules, pinInRAM= True, budgets=budgets, info = info, quantizeTransformer= quantizeTransformer )
        elif profile_no == profile_type.LowRAM_HighVRAM_Medium:
            info = "You have chosen a Medium speed profile that requires at least 32 GB of RAM and 24 GB of VRAM."
            return offload.all(pipe_or_dict_of_modules, pinInRAM= "transformer", modelsToQuantize= extra_mod_to_quantize ,  info = info, quantizeTransformer= quantizeTransformer)
        elif profile_no == profile_type.LowRAM_LowVRAM_Slow:
            info = "You have chosen the Slow profile that requires at least 32 GB of RAM and 12 GB of VRAM."
            return offload.all(pipe_or_dict_of_modules, pinInRAM= "transformer", modelsToQuantize= extra_mod_to_quantize ,  budgets=budgets, info = info, quantizeTransformer= quantizeTransformer)
        elif profile_no == profile_type.VerylowRAM_LowVRAM_Slowest:
            budgets["transformer"] = 400
            info = "You have chosen the Slowest profile that requires at least 24 GB of RAM and 10 GB of VRAM."
            return offload.all(pipe_or_dict_of_modules, pinInRAM= False, modelsToQuantize= extra_mod_to_quantize ,  budgets=budgets, info = info, quantizeTransformer= quantizeTransformer)
        else:
            raise("Unknown profile")

