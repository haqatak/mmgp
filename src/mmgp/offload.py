# ------------------ Memory Management 3.1.4-1519 for the GPU Poor by DeepBeepMeep (mmgp)------------------
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
# Make sure that the pipeline explictly loads the models in the CPU device 
#   for instance: pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cpu")
# For a quick setup, you may want to choose between 5 profiles depending on your hardware, for instance:
#   from mmgp import offload, profile_type
#   offload.profile(pipe, profile_type.HighRAM_LowVRAM_Fast)
# Alternatively you may want to your own parameters, for instance:
#   from mmgp import offload
#   offload.all(pipe, pinToMemory=true, extraModelsToQuantize = ["text_encoder_2"] )
# The 'transformer' model that contains usually the video or image generator is quantized on the fly by default to 8 bits so that it can fit into 24 GB of VRAM. 
# You can prevent the transformer quantization by adding the parameter quantizeTransformer = False
# If you want to save time on disk and reduce the loading time, you may want to load directly a prequantized model. In that case you need to set the option quantizeTransformer to False to turn off on the fly quantization.
# You can specify a list of additional models string ids to quantize (for instance the text_encoder) using the optional argument extraModelsToQuantize. This may be useful if you have less than 48 GB of RAM.
# Note that there is little advantage on the GPU / VRAM side to quantize text encoders as their inputs are usually quite light. 
# Conversely if you have more than 48GB RAM you may want to enable RAM pinning with the option pinnedMemory = True. You will get in return super fast loading / unloading of models
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
import gc
import time
import functools
import sys
import os
import json
import psutil
try:    
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters, get_peft_kwargs
except:
    set_weights_and_activate_adapters = None
    get_peft_kwargs = None
    pass
try:    
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
except:
    inject_adapter_in_model = None
    pass

from mmgp import safetensors2
from mmgp import profile_type

from optimum.quanto import freeze,  qfloat8, qint4 , qint8, quantize, QModuleMixin, QTensor,  quantize_module, register_qmodule

# support for Embedding module quantization that is not supported by default by quanto
@register_qmodule(torch.nn.Embedding)
class QEmbedding(QModuleMixin, torch.nn.Embedding):
    @classmethod
    def qcreate(cls, module, weights, activations = None, optimizer = None, device = None):
        module.bias = None
        return cls( module.num_embeddings, module.embedding_dim, module.padding_idx , module.max_norm, module.norm_type, module.scale_grad_by_freq, module.sparse, dtype=module.weight.dtype, device=device, weights=weights,
                    activations=activations, optimizer=optimizer, quantize_input=True)      
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding( input, self.qweight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse )


shared_state = {}

mmm = safetensors2.mmm

default_verboseLevel = 1

ONE_MB =  1048576
sizeofbfloat16 = torch.bfloat16.itemsize
sizeofint8 = torch.int8.itemsize
total_pinned_bytes = 0
physical_memory= psutil.virtual_memory().total

HEADER = '\033[95m'
ENDC = '\033[0m'
BOLD ='\033[1m'
UNBOLD ='\033[0m'

class clock:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    @classmethod
    def start(cls):
        self = cls()        
        self.start_time =time.time()
        return self        

    def stop(self):
        self.stop_time =time.time()  

    def time_gap(self):
        return self.stop_time - self.start_time
    
    def format_time_gap(self):
        return f"{self.stop_time - self.start_time:.2f}s"



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
def _get_module_name(v):
    return v.__module__.lower()


def _compute_verbose_level(level):
    if level <0:        
        level = safetensors2.verboseLevel = default_verboseLevel
    safetensors2.verboseLevel = level
    return level

def _get_perc_reserved_mem_max(perc_reserved_mem_max):
    if perc_reserved_mem_max<=0:             
        perc_reserved_mem_max = 0.40 if os.name == 'nt' else 0.5        
    return  perc_reserved_mem_max 

def _detect_main_towers(model, min_floors = 5):
    cur_blocks_prefix = None
    towers_modules= []
    towers_names= []

    floors_modules= []
    tower_name = None


    for submodule_name, submodule in model.named_modules():  

        if submodule_name=='':
            continue

        if cur_blocks_prefix != None:
            if submodule_name.startswith(cur_blocks_prefix):
                depth_prefix = cur_blocks_prefix.split(".")
                depth_name = submodule_name.split(".")
                level  =  depth_name[len(depth_prefix)-1]                        
                pre , num = _extract_num_from_str(level)

                if num != cur_blocks_seq: 
                    floors_modules.append(submodule)

                cur_blocks_seq = num
            else:
                if len(floors_modules) >= min_floors:
                    towers_modules += floors_modules
                    towers_names.append(tower_name)
                tower_name = None
                floors_modules= []
                cur_blocks_prefix, cur_blocks_seq = None, -1

        if cur_blocks_prefix == None:
            pre , num = _extract_num_from_str(submodule_name)
            if isinstance(submodule, (torch.nn.ModuleList)):  
                cur_blocks_prefix, cur_blocks_seq = pre + ".",  -1
                tower_name = submodule_name + "." 
            elif num >=0:
                cur_blocks_prefix, cur_blocks_seq = pre, num
                tower_name = submodule_name[ :-1]  
                floors_modules.append(submodule)

    if len(floors_modules) >= min_floors:
        towers_modules += floors_modules
        towers_names.append(tower_name)

    return towers_names, towers_modules



def _get_model(model_path):
    if os.path.isfile(model_path):
        return model_path
    
    from pathlib import Path
    _path = Path(model_path).parts
    _filename = _path[-1]
    _path = _path[:-1]
    if len(_path)<=1:
        raise("file not found")
    else:
        try:
            from huggingface_hub import  hf_hub_download #snapshot_download,    
            repoId=  os.path.join(*_path[0:2] ).replace("\\", "/")

            if len(_path) > 2:
                _subfolder = os.path.join(*_path[2:] )
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename,  subfolder=_subfolder)
            else:
                model_path = hf_hub_download(repo_id=repoId,  filename=_filename)
        except:
           model_path = None 
    return model_path



def _remove_model_wrapper(model):
    if not model._modules is None:
        if len(model._modules)!=1:
            return model
    sub_module = model._modules[next(iter(model._modules))]
    if hasattr(sub_module,"config") or hasattr(sub_module,"base_model"):
        return sub_module
    return model  

 

def _move_to_pinned_tensor(source_tensor, big_tensor, offset, length):
    dtype= source_tensor.dtype
    shape = source_tensor.shape
    if len(shape) == 0:
        return source_tensor
    else:                
        t = source_tensor.view(torch.uint8)
        t = torch.reshape(t, (length,))
        # magic swap !
        big_tensor[offset: offset + length] = t 
        t = big_tensor[offset: offset + length]
        t = t.view(dtype)
        t = torch.reshape(t, shape)
        assert t.is_pinned()
    return t

def _safetensors_load_file(file_path):
    from collections import OrderedDict
    sd = OrderedDict()    

    with safetensors2.safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
        metadata = f.metadata()

    return sd, metadata

def _force_load_buffer(p):
    # To do : check if buffer was persistent and transfer state, or maybe swap keep already this property ?
    q = torch.nn.Buffer(p + 0)
    torch.utils.swap_tensors(p, q)
    del q

def _force_load_parameter(p):
    q = torch.nn.Parameter(p + 0)
    torch.utils.swap_tensors(p, q)
    del q

def _get_tensor_ref(p):
    if isinstance(p, QTensor):
        if p._qtype == qint4:
            return p._data._data.data_ptr()
        else:
            return p._data.data_ptr()
    else:                
        return p.data_ptr()


def _pin_to_memory(model, model_id, partialPinning = False, verboseLevel = 1):
    if partialPinning:
        towers_names, _ = _detect_main_towers(model)


    BIG_TENSOR_MAX_SIZE = 2**28 # 256 MB
    current_big_tensor_size = 0
    big_tensor_no  = 0
    big_tensors_sizes = []
    tensor_map_indexes = []
    total_tensor_bytes = 0

    params_dict = {} #  OrderedDict
    for k, sub_module in model.named_modules():
        include = True
        if partialPinning:
            include = any(k.startswith(pre) for pre in towers_names) if partialPinning else True
        if include:
            params_dict.update( { k + '.' + n : (p,  False) for n, p in sub_module.named_parameters(recurse=False) }  )
            params_dict.update( { k + '.' + n : (b,  True) for n, b in sub_module.named_buffers(recurse=False) }  )

    if  verboseLevel>=1 :
        if partialPinning:
            if len(params_dict) == 0:
                print(f"Unable to apply Partial of '{model_id}' as no isolated main structures were found")
            else:
                print(f"Partial pinning of data of '{model_id}' to reserved RAM")
        else:            
            print(f"Pinning data of '{model_id}' to reserved RAM")

    if partialPinning and len(params_dict) == 0:
        return

    ref_cache = {}
    tied_weights = {}
    for n, (p, _) in params_dict.items():
        ref = _get_tensor_ref(p)
        match = ref_cache.get(ref, None)
        if match != None:
            match_name, match_size = match
            if verboseLevel >=1:
                print(f"Tied weights of {match_size/ONE_MB:0.2f} MB detected: {match_name} <-> {n}")
            tied_weights[n] = match_name
        else:
            if isinstance(p, QTensor):
                if p._qtype == qint4:
                    if hasattr(p,"_scale_shift"):
                        length = torch.numel(p._data._data) * p._data._data.element_size() + torch.numel(p._scale_shift) * p._scale_shift.element_size() 
                    else:
                        length = torch.numel(p._data._data) * p._data._data.element_size() + torch.numel(p._scale) * p._scale.element_size() + torch.numel(p._shift) * p._shift.element_size()                     
                else:
                    length = torch.numel(p._data) * p._data.element_size() + torch.numel(p._scale) * p._scale.element_size() 
            else:
                length = torch.numel(p.data) * p.data.element_size() 

            ref_cache[ref] = (n, length)
            if current_big_tensor_size + length > BIG_TENSOR_MAX_SIZE:
                big_tensors_sizes.append(current_big_tensor_size)
                current_big_tensor_size = 0
                big_tensor_no += 1


            itemsize = p.data.dtype.itemsize
            if current_big_tensor_size % itemsize:
                current_big_tensor_size += itemsize - current_big_tensor_size % itemsize
            tensor_map_indexes.append((big_tensor_no, current_big_tensor_size, length  ))
            current_big_tensor_size += length

            total_tensor_bytes += length
  
    big_tensors_sizes.append(current_big_tensor_size)

    big_tensors = []
    last_big_tensor = 0
    total = 0
    


    for size in big_tensors_sizes:
        try:
            current_big_tensor = torch.empty( size, dtype= torch.uint8, pin_memory=True, device="cpu")
            big_tensors.append(current_big_tensor)
        except:
            print(f"Unable to pin more tensors for this model as the maximum reservable memory has been reached ({total/ONE_MB:.2f})")
            break

        last_big_tensor += 1
        total += size


    gc.collect()

        
    tensor_no = 0
    # prev_big_tensor = 0
    for n, (p, is_buffer) in params_dict.items():
        if n in tied_weights:
            if isinstance(p, QTensor):
                if p._qtype == qint4:                
                    assert p._data._data.data.is_pinned()
                else:
                    assert p._data.is_pinned()
            else:
                assert p.data.is_pinned()
        else:
            big_tensor_no, offset, length = tensor_map_indexes[tensor_no]
            # if big_tensor_no != prev_big_tensor:
            #     gc.collect()
            #     prev_big_tensor = big_tensor_no
            # match_param, match_isbuffer = tied_weights.get(n, (None, False))
            # if match_param != None:

            if big_tensor_no>=0 and big_tensor_no < last_big_tensor:
                current_big_tensor = big_tensors[big_tensor_no]
                if is_buffer :
                    _force_load_buffer(p) # otherwise potential memory leak
                if isinstance(p, QTensor):
                    if p._qtype == qint4:
                        length1 = torch.numel(p._data._data) * p._data._data.element_size()
                        p._data._data =  _move_to_pinned_tensor(p._data._data, current_big_tensor, offset, length1)
                        if hasattr(p,"_scale_shift"):
                            length2 = torch.numel(p._scale_shift) * p._scale_shift.element_size() 
                            p._scale_shift = _move_to_pinned_tensor(p._scale_shift, current_big_tensor, offset + length1, length2)
                        else:
                            length2 = torch.numel(p._scale) * p._scale.element_size() 
                            p._scale = _move_to_pinned_tensor(p._scale, current_big_tensor, offset + length1, length2)
                            length3 = torch.numel(p._shift) * p._shift.element_size() 
                            p._shift = _move_to_pinned_tensor(p._shift, current_big_tensor, offset + length1 + length2, length3)
                    else:
                        length1 = torch.numel(p._data) * p._data.element_size() 
                        p._data = _move_to_pinned_tensor(p._data, current_big_tensor, offset, length1)
                        length2 = torch.numel(p._scale) * p._scale.element_size() 
                        p._scale = _move_to_pinned_tensor(p._scale, current_big_tensor, offset + length1, length2)
                else:
                    length = torch.numel(p.data) * p.data.element_size() 
                    p.data = _move_to_pinned_tensor(p.data, current_big_tensor, offset, length)

            tensor_no += 1
        del p
    global total_pinned_bytes
    total_pinned_bytes += total
    del params_dict
    gc.collect()

    if verboseLevel >=1:
        if partialPinning:        
            print(f"The model was partially pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")
        else:
            print(f"The whole model was pinned to reserved RAM: {last_big_tensor} large blocks spread across {total/ONE_MB:.2f} MB")

    model._already_pinned = True


    return 
welcome_displayed = False

def _welcome():
    global welcome_displayed
    if welcome_displayed:
         return 
    welcome_displayed = True
    print(f"{BOLD}{HEADER}************ Memory Management for the GPU Poor (mmgp 3.1.4-151) by DeepBeepMeep ************{ENDC}{UNBOLD}")

def _extract_num_from_str(num_in_str):
    size = len(num_in_str)
    for i in range(size):
        if not num_in_str[-i-1:].isnumeric():
            if i == 0:
                return num_in_str, -1
            else:             
                return num_in_str[: -i],  int(num_in_str[-i:])                    
    return  "", -1 if size == 0 else int(num_in_str)

def  _quantize_dirty_hack(model):
    # dirty hack: add a hook on state_dict() to return a fake non quantized state_dict if called by Lora Diffusers initialization functions
    setattr( model, "_real_state_dict", model.state_dict)
    from collections import OrderedDict
    import traceback

    def state_dict_for_lora(self):
        real_sd = self._real_state_dict()
        fakeit = False
        stack = traceback.extract_stack(f=None, limit=5)
        for frame in stack:
            if "_lora_" in frame.name:
                fakeit = True
                break

        if not fakeit:
            return real_sd
        sd = OrderedDict()
        for k in real_sd:
            v = real_sd[k]
            if k.endswith("._data"):
                k = k[:len(k)-6]
            sd[k] = v
        return sd

    setattr(model, "state_dict", functools.update_wrapper(functools.partial(state_dict_for_lora, model), model.state_dict) )

def _quantization_map(model):
    from optimum.quanto import quantization_map
    return quantization_map(model)

def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)

def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights = None,
    activations = None,
    optimizer = None,
):
    
    qmodule = quantize_module(module, weights=weights, activations=activations, optimizer=optimizer)
    if qmodule is not None:
        _set_module_by_name(model, name, qmodule)
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param

def _requantize(model: torch.nn.Module, state_dict: dict, quantization_map: dict):
    # change dtype of current meta model parameters because 'requantize' won't update the dtype on non quantized parameters
    for k, p in model.named_parameters():
        if not k in quantization_map and k in state_dict:
            p_in_file = state_dict[k] 
            if p.data.dtype != p_in_file.data.dtype:
                p.data = p.data.to(p_in_file.data.dtype)

    # rebuild quanto objects
    for name, m in model.named_modules():
        qconfig = quantization_map.get(name, None)
        if qconfig is not None:
            weights = qconfig["weights"]
            if weights == "none":
                weights = None
            activations = qconfig["activations"]
            if activations == "none":
                activations = None
            _quantize_submodule(model, name, m, weights=weights, activations=activations)

    model._quanto_map = quantization_map

    _quantize_dirty_hack(model)



def _quantize(model_to_quantize, weights=qint8, verboseLevel = 1, threshold = 1000000000, model_id = 'Unknown'):
    
    total_size =0
    total_excluded = 0
    exclude_list = []
    submodule_size = 0
    submodule_names = []
    cur_blocks_prefix = None
    prev_blocks_prefix = None

    if hasattr(model_to_quantize, "_quanto_map"):
        for k, entry in model_to_quantize._quanto_map.items():
            weights  =  entry["weights"]
            print(f"Model '{model_id}' is already quantized to format '{weights}'")
            return False
        print(f"Model '{model_id}' is already quantized")
        return False

    print(f"Quantization of model '{model_id}' started to format '{weights}'")

    tower_names ,_  = _detect_main_towers(model_to_quantize)
    tower_names = [ n[:-1] for n in tower_names]


    cache_ref = {}
    tied_weights= {}

    for submodule_name, submodule in model_to_quantize.named_modules():  
        if isinstance(submodule, QModuleMixin):
            if verboseLevel>=1:
                print("No quantization to do as model is already quantized")
            return False

        size = 0
        for n, p in submodule.named_parameters(recurse = False):
            ref = _get_tensor_ref(p)
            match = cache_ref.get(ref, None)
            if match != None:
                tied_weights[submodule_name]=  (n, ) + match 
            else:
                cache_ref[ref] = (submodule_name, n)
                size  += torch.numel(p.data) * sizeofbfloat16

        for p in submodule.buffers(recurse=False):
            size  += torch.numel(p.data) * sizeofbfloat16



        if not any(submodule_name.startswith(pre) for pre in tower_names):
            flush = False
            if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
                if cur_blocks_prefix == None:
                    cur_blocks_prefix = submodule_name + "."
                    flush = True                    
                else:
                    if not submodule_name.startswith(cur_blocks_prefix):
                        cur_blocks_prefix = submodule_name + "."
                        flush = True                    
            else:                
                if cur_blocks_prefix is not None:
                    #if not cur_blocks_prefix == submodule_name[0:len(cur_blocks_prefix)]:
                    if not submodule_name.startswith(cur_blocks_prefix):
                        cur_blocks_prefix = None 
                        flush = True                    

            if flush :
                if submodule_size <= threshold :
                    exclude_list += submodule_names
                    if verboseLevel >=2:
                        print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
                    total_excluded += submodule_size

                submodule_size = 0
                submodule_names = []
            prev_blocks_prefix = cur_blocks_prefix
            submodule_size += size
            submodule_names.append(submodule_name)
        total_size += size

    if submodule_size >0 and submodule_size <= threshold :
        exclude_list += submodule_names
        if verboseLevel >=2:
            print(f"Excluded size {submodule_size/ONE_MB:.1f} MB: {prev_blocks_prefix} : {submodule_names}")
        total_excluded += submodule_size


    perc_excluded =total_excluded/ total_size if total_size >0 else 1
    if verboseLevel >=2:
        if total_excluded == 0:
            print(f"Can't find any module to exclude from quantization, full model ({total_size/ONE_MB:.1f} MB) will be quantized")
        else:
            print(f"Total Excluded {total_excluded/ONE_MB:.1f} MB of {total_size/ONE_MB:.1f} that is {perc_excluded*100:.2f}%")
    if perc_excluded >= 0.10:
        if verboseLevel >=2:
            print(f"Too many modules are excluded, there is something wrong with the selection, switch back to full quantization.")
        exclude_list = None


    exclude_list += list(tied_weights) 
    quantize(model_to_quantize, weights= weights, exclude= exclude_list)


    # quantize(model_to_quantize,weights, include= [ "*1.block.attn.to_out*"]) #" 

    # for name, m in model_to_quantize.named_modules():
    #     if exclude_list is None or not any( name == module_name for module_name in exclude_list):
    #         _quantize_submodule(model_to_quantize, name, m, weights=weights, activations=None, optimizer=None)


    # force to read non quantized parameters so that their lazy tensors and corresponding mmap are released
    # otherwise we may end up keeping in memory both the quantized and the non quantize model
    named_modules = {n:m for n,m in model_to_quantize.named_modules()}
    for module_name, module in named_modules.items():
        # do not read quantized weights (detected them directly or behind an adapter)
        if isinstance(module, QModuleMixin) or hasattr(module, "base_layer") and  isinstance(module.base_layer, QModuleMixin): 
            if hasattr(module, "bias") and module.bias is not None:
                _force_load_parameter(module.bias)
        else:
            tied_w = tied_weights.get(module_name, None)
            for n, p in module.named_parameters(recurse = False):
                if tied_w != None and n == tied_w[0]:
                    if isinstance( named_modules[tied_w[1]], QModuleMixin) :
                        setattr(module, n, None) # release refs of tied weights if source is going to be quantized
                    # otherwise don't force load as it will be loaded in the source anyway
                else:
                    _force_load_parameter(p)
                del p #  del p if not it will still contain a ref to a tensor when leaving the loop
        for b in module.buffers(recurse = False):
            _force_load_buffer(b) 
            del b


    freeze(model_to_quantize)
    torch.cuda.empty_cache()
    gc.collect()       

    for tied_module, (tied_weight, src_module, src_weight) in tied_weights.items():  
        p = getattr(named_modules[src_module], src_weight)
        if isinstance(p, QTensor):
            setattr(named_modules[tied_module], tied_weight, p ) # copy refs to quantized sources

    del named_modules

    quantization_map = _quantization_map(model_to_quantize)

    model_to_quantize._quanto_map = quantization_map

    if hasattr(model_to_quantize, "_already_pinned"):
        delattr(model_to_quantize, "_already_pinned")

    _quantize_dirty_hack(model_to_quantize)

    print(f"Quantization of model '{model_id}' done")

    return True

def _lora_linear_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        base_weight = self.base_layer.weight
        if base_weight.shape[-1] < x.shape[-2]: # sum base weight and lora matrices instead of applying input on each sub lora matrice if input is too large. This will save a lot VRAM and compute
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                if self.use_dora[active_adapter]:
                    raise Exception("Dora not yet supported by mmgp")
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                scaling = self.scaling[active_adapter]
                lora_A_weight = lora_A.weight
                lora_B_weight = lora_B.weight
                lora_BA = lora_B_weight @ lora_A_weight   
                base_weight += scaling * lora_BA

            result = torch.nn.functional.linear(x, base_weight, bias=self.base_layer.bias)
            torch_result_dtype = result.dtype

        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            x = x.to(torch.bfloat16)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    y = lora_A(x)
                    y = lora_B(y)
                    y*= scaling
                    result+= y 
                    del lora_A, lora_B, y
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

        result = result.to(torch_result_dtype)
        return result
    
def load_loras_into_model(model, lora_path, lora_multi = None, activate_all_loras = True, verboseLevel = -1,):
    verboseLevel = _compute_verbose_level(verboseLevel)

    if inject_adapter_in_model == None or set_weights_and_activate_adapters == None or  get_peft_kwargs == None:
        raise Exception("Unable to load Lora, missing 'peft' and / or 'diffusers' modules")

    from peft.tuners.lora import Linear
    Linear.forward = _lora_linear_forward

    if not isinstance(lora_path, list):
        lora_path = [lora_path]
    
    if lora_multi is None:
        lora_multi = [1. for _ in lora_path]

    for i, path in enumerate(lora_path):
        adapter_name = str(i)




        state_dict = safetensors2.torch_load_file(path)
        
        keys = list(state_dict.keys())
        if len(keys) == 0:
            raise Exception(f"Empty Lora '{path}'")


        network_alphas = {}
        for k in keys:
            if "alpha" in k:
                alpha_value = state_dict.pop(k)
                if not ( (torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)) or isinstance(
                    alpha_value, float
                )):
                    network_alphas[k] =  torch.tensor( float(alpha_value.item() ) )

        pos = keys[0].find(".")
        prefix = keys[0][0:pos]
        if not any( prefix.startswith(some_prefix) for some_prefix in ["diffusion_model", "transformer"]): 
            msg = f"No compatible weight was found in Lora file '{path}'. Please check that it is compatible with the Diffusers format."
            raise Exception(msg)

        transformer = model

        transformer_keys = [k for k in keys if k.startswith(prefix)]
        state_dict = {
            k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in transformer_keys
        }

        sd_keys = state_dict.keys()
        if len(sd_keys) == 0:
            print(f"No compatible weight was found in Lora file '{path}'. Please check that it is compatible with the Diffusers format.")
            return

        # is_correct_format = all("lora" in key for key in state_dict.keys())

        # check with first key if is not in peft format
        # first_key = next(iter(state_dict.keys()))
        # if "lora_A" not in first_key:
        #     state_dict = convert_unet_state_dict_to_peft(state_dict)

        if adapter_name in getattr(transformer, "peft_config", {}):
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
            )

        rank = {}
        for key, val in state_dict.items():
            if "lora_B" in key:
                rank[key] = val.shape[1]

        if network_alphas is not None and len(network_alphas) >= 1:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix]
            network_alphas = {k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys}

        lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)
        
        lora_config = LoraConfig(**lora_config_kwargs)
        peft_kwargs = {}        
        peft_kwargs["low_cpu_mem_usage"] = True
        inject_adapter_in_model(lora_config, model, adapter_name=adapter_name, **peft_kwargs)

        incompatible_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name, **peft_kwargs)

        warn_msg = ""
        if incompatible_keys is not None:
            # Check only for unexpected keys.
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                pass
        if verboseLevel >=1:
            print(f"Lora '{path}' was loaded in model '{_get_module_name(model)}'")
    if activate_all_loras:
        set_weights_and_activate_adapters(model,[ str(i) for i in range(len(lora_multi))], lora_multi)

def activate_loras(model, lora_nos, lora_multi = None ):
    if not isinstance(lora_nos, list):
        lora_nos = [lora_nos]
    lora_nos = [str(l) for l in lora_nos]
    if lora_multi is None:
        lora_multi = [1. for _ in lora_nos]
    set_weights_and_activate_adapters(model, lora_nos, lora_multi)


def move_loras_to_device(model, device="cpu" ):
    if hasattr( model, "_lora_loadable_modules"):
        for k in model._lora_loadable_modules:
            move_loras_to_device(getattr(model,k), device)
        return
    
    for k, m in model.named_modules():
        if ".lora_" in k:
            m.to(device)

def fast_load_transformers_model(model_path: str, do_quantize = False, quantizationType =  qint8, pinToMemory = False, partialPinning = False, verboseLevel = -1):
    """
    quick version of .LoadfromPretrained of  the transformers library
    used to build a model and load the corresponding weights (quantized or not)
    """       

    
    import os.path
    from accelerate import init_empty_weights
 
    if not (model_path.endswith(".sft") or model_path.endswith(".safetensors")):
        raise Exception("full model path to file expected")

    model_path = _get_model(model_path)
    verboseLevel = _compute_verbose_level(verboseLevel)

    with safetensors2.safe_open(model_path) as f:
        metadata = f.metadata() 

    if metadata is None:
        transformer_config = None
    else:
        transformer_config = metadata.get("config", None)

    if transformer_config == None:
        config_fullpath =  os.path.join(os.path.dirname(model_path), "config.json")

        if not os.path.isfile(config_fullpath):
            raise Exception("a 'config.json' that describes the model is required in the directory of the model or inside the safetensor file")

        with open(config_fullpath, "r", encoding="utf-8") as reader:
            text = reader.read()
        transformer_config= json.loads(text)


    if "architectures" in transformer_config: 
        architectures = transformer_config["architectures"]
        class_name = architectures[0] 

        module = __import__("transformers")
        map = {  "T5WithLMHeadModel" : "T5EncoderModel"}
        class_name = map.get(class_name, class_name)
        transfomer_class = getattr(module, class_name)
        from transformers import AutoConfig

        import tempfile
        with tempfile.NamedTemporaryFile("w", delete = False,  encoding ="utf-8") as fp: 
            fp.write(json.dumps(transformer_config))
            fp.close()
            config_obj = AutoConfig.from_pretrained(fp.name)     
        os.remove(fp.name)

        #needed to keep inits of non persistent buffers
        with init_empty_weights():
            model = transfomer_class(config_obj)
                
        model = model.base_model

    elif "_class_name" in transformer_config:
        class_name = transformer_config["_class_name"]

        module = __import__("diffusers")
        transfomer_class = getattr(module, class_name)

        with init_empty_weights():
            model = transfomer_class.from_config(transformer_config)


    torch.set_default_device('cpu')

    model._config = transformer_config
            
    load_model_data(model,model_path, do_quantize = do_quantize, quantizationType = quantizationType, pinToMemory= pinToMemory, partialPinning= partialPinning, verboseLevel=verboseLevel )

    return model



def load_model_data(model, file_path: str, do_quantize = False, quantizationType = qint8, pinToMemory = False, partialPinning = False, verboseLevel = -1):
    """
    Load a model, detect if it has been previously quantized using quanto and do the extra setup if necessary
    """

    file_path = _get_model(file_path)
    verboseLevel = _compute_verbose_level(verboseLevel)

    model = _remove_model_wrapper(model)
    if not (".safetensors" in file_path or ".sft" in file_path): 
        if pinToMemory:
            raise Exception("Pinning to memory while loading only supported for safe tensors files")
        state_dict = torch.load(file_path, weights_only=True)
        if "module" in state_dict:
            state_dict = state_dict["module"]
    else:
        state_dict, metadata = _safetensors_load_file(file_path)
            
        if metadata is None:
            quantization_map = None
            tied_weights_map = None
        else:
            quantization_map = metadata.get("quantization_map", None)
            config = metadata.get("config", None)
            if config is not None:
                model._config = config

            tied_weights_map = metadata.get("tied_weights_map", None)
            if tied_weights_map != None:
                for name, tied_weights_list in tied_weights_map.items():
                    mapped_weight = state_dict[name]
                    for tied_weights in tied_weights_list:
                        state_dict[tied_weights] = mapped_weight



        if quantization_map is None:
            pos = str.rfind(file_path, ".")
            if pos > 0:
                quantization_map_path = file_path[:pos]
            quantization_map_path += "_map.json"

            if os.path.isfile(quantization_map_path):
                with open(quantization_map_path, 'r') as f:
                    quantization_map = json.load(f)



        if quantization_map is None :
            if "quanto" in file_path and not do_quantize:
                print("Model seems to be quantized by quanto but no quantization map was found whether inside the model or in a separate '{file_path[:json]}_map.json' file")
        else:
            _requantize(model, state_dict, quantization_map)    

    missing_keys , unexpected_keys = model.load_state_dict(state_dict, False,  assign = True )
    # if len(missing_keys) > 0:
    #     sd_crap = { k : None for k in missing_keys}
    #     missing_keys , unexpected_keys = model.load_state_dict(sd_crap, strict =False,  assign = True )
    del state_dict

    for k,p in model.named_parameters():
        if p.is_meta:
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since parameter '{k}' has no data"
            raise Exception(txt)
    for k,b in model.named_buffers():
        if b.is_meta:
            txt  = f"Incompatible State Dictionary or 'Init_Empty_Weights' not set since buffer '{k}' has no data"
            raise Exception(txt)

    if do_quantize:
        if quantization_map is None:
            if _quantize(model, quantizationType, verboseLevel=verboseLevel, model_id=file_path):
                quantization_map = model._quanto_map  
        else:
            if verboseLevel >=1:
                print("Model already quantized")

    if pinToMemory:
        _pin_to_memory(model, file_path, partialPinning = partialPinning, verboseLevel = verboseLevel)

    return

def save_model(model, file_path, do_quantize = False, quantizationType = qint8, verboseLevel = -1, config_file_path = None ):
    """save the weights of a model and quantize them if requested
    These weights can be loaded again using 'load_model_data'
    """       
    
    config = None
    extra_meta = None
    verboseLevel = _compute_verbose_level(verboseLevel)
    if config_file_path !=None:
        with open(config_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
            config= json.loads(text)
    elif hasattr(model, "_config"):
        config = model._config
    elif hasattr(model, "config"):
        config_fullpath = None
        config_obj = getattr(model,"config")
        config_path = getattr(config_obj,"_name_or_path", None)
        if config_path != None:
            config_fullpath = os.path.join(config_path, "config.json")      
            config_fullpath = _get_model(config_fullpath)

            # if not os.path.isfile(config_fullpath):
            #     config_fullpath = None
        if config_fullpath is None:                            
            config_fullpath =  os.path.join(os.path.dirname(file_path), "config.json")
        if os.path.isfile(config_fullpath):
            with open(config_fullpath, "r", encoding="utf-8") as reader:
                text = reader.read()
                config= json.loads(text)

    if do_quantize:
        _quantize(model, weights=quantizationType, model_id=file_path)
    
    quantization_map = getattr(model, "_quanto_map", None)

    from collections import OrderedDict

    cache_ref = {}
    tied_weights_map = {}
    sd = model.state_dict()
    out_sd = OrderedDict()


    for name, weight  in sd.items():
        ref = _get_tensor_ref(weight)
        match = cache_ref.get(ref, None)
        if match != None:
            tied_list = tied_weights_map.get(match, [])
            tied_list.append(name)
            tied_weights_map[match] = tied_list 
        else:
            out_sd[name] = weight 
            cache_ref[ref] = name

    if len(tied_weights_map) > 0:
        extra_meta = { "tied_weights_map" : tied_weights_map }

    if verboseLevel >=1:
        print(f"Saving file '{file_path}")

    safetensors2.torch_write_file(out_sd,  file_path , quantization_map = quantization_map, config = config, extra_meta= extra_meta)
    if verboseLevel >=1:
        print(f"File '{file_path}' saved")


def extract_models(obj = None, prefix = None):
    if isinstance(obj, str): # for compatibility as the two args were switched
        bkp = prefix
        prefix = obj
        obj = bkp

    pipe = {}
    if obj == None:
        raise Exception("an object to analyze must be provided")
    if prefix==None or len(prefix)==0:
        prefix = ""
    elif prefix[ -1:] != "/":
        prefix  + "/"        
    
    for name in dir(obj):            
        element = getattr(obj,name)
        if name  in ("pipeline", "pipe"):
            pipeline = element
            if  hasattr(pipeline , "components") and isinstance(pipeline.components, dict):
                for k, model in pipeline.components.items():
                    if model != None:
                        pipe[prefix  + k ] = model
        elif isinstance(element, torch.nn.Module) and name!="base_model": 
            if prefix + name in pipe:
                pipe[prefix + "_" + name ] = element
            else:
                pipe[prefix + name ] = element
        elif isinstance(element, dict):
            for k, element in element.items():
                if  hasattr(element , "pipeline"):
                    pipe.update( extract_models(prefix + k,element ))


    return pipe

def get_model_name(model):
    return model.name

class HfHook:
    def __init__(self):
        self.execution_device = "cuda"

    def detach_hook(self, module):
        pass

last_offload_obj = None
class offload:
    def __init__(self):
        self.active_models = []
        self.active_models_ids = []
        self.active_subcaches = {}        
        self.models = {}
        self.cotenants_map = { 
                            "text_encoder": ["vae", "text_encoder_2"],
                            "text_encoder_2": ["vae", "text_encoder"],                             
                        }
        self.verboseLevel = 0
        self.blocks_of_modules = {}
        self.blocks_of_modules_sizes = {}
        self.anyCompiledModule = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check =0
        self.loaded_blocks = {}
        self.prev_blocks_names = {}
        self.next_blocks_names = {}
        self.preloaded_blocks_per_model = {}
        self.default_stream = torch.cuda.default_stream(torch.device("cuda")) # torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        self.async_transfers = False
        self.parameters_ref  = {} 
        global last_offload_obj
        last_offload_obj = self

        
    def add_module_to_blocks(self, model_id, blocks_name, submodule, prev_block_name, submodule_name):

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
        bef = blocks_params_size
        for k,p in submodule.named_parameters(recurse=False):
            param_size = 0
            ref = _get_tensor_ref(p)
            tied_param =  self.parameters_ref.get(ref, None)

            if isinstance(p, QTensor):
                blocks_params.append( (submodule, k, p, False, tied_param ) )

                if p._qtype == qint4:
                    if hasattr(p,"_scale_shift"):
                        param_size += torch.numel(p._scale_shift) * p._scale_shift.element_size()
                        param_size += torch.numel(p._data._data) * p._data._data.element_size()
                    else:
                        param_size += torch.numel(p._scale) * p._scale.element_size()
                        param_size += torch.numel(p._shift) * p._shift.element_size()
                        param_size += torch.numel(p._data._data) * p._data._data.element_size()
                else:
                    param_size += torch.numel(p._scale) * p._scale.element_size()
                    param_size += torch.numel(p._data) * p._data.element_size()
            else:
                blocks_params.append( (submodule, k, p, False, tied_param) )
                param_size += torch.numel(p.data) * p.data.element_size()


            if tied_param == None:
                blocks_params_size +=  param_size
                self.parameters_ref[ref] = (submodule, k)

        for k, p in submodule.named_buffers(recurse=False):
            blocks_params.append( (submodule, k, p, True, None) )
            blocks_params_size += p.data.nbytes

        aft = blocks_params_size

        # if blocks_name is None:
        #     print(f"Default: {model_id}/{submodule_name} : {(aft-bef)/ONE_MB:0.2f} MB")
        #     pass


        self.blocks_of_modules_sizes[entry_name] = blocks_params_size


        return blocks_params_size


    def can_model_be_cotenant(self, model_id):
        potential_cotenants= self.cotenants_map.get(model_id, None)
        if potential_cotenants is None: 
            return False
        for existing_cotenant in self.active_models_ids:
            if existing_cotenant not in potential_cotenants: 
                return False    
        return True

    @torch.compiler.disable()
    def gpu_load_blocks(self, model_id, blocks_name, preload = False):
        # cl = clock.start()


        entry_name = model_id if blocks_name is None else model_id + "/" + blocks_name
        
        def cpu_to_gpu(stream_to_use, blocks_params): #, record_for_stream = None
            with torch.cuda.stream(stream_to_use):
                for param in blocks_params:
                    parent_module, n, p, is_buffer, tied_param  = param
                    if tied_param != None:
                        tied_p = getattr( tied_param[0], tied_param[1]) 
                        if tied_p.is_cuda:
                            setattr(parent_module, n , tied_p)
                            continue

                    q = p.to("cuda", non_blocking=True)
                    if is_buffer:
                        q = torch.nn.Buffer(q)
                    else:
                        q = torch.nn.Parameter(q , requires_grad=False)
                    setattr(parent_module, n , q)

                    if tied_param != None:
                        setattr( tied_param[0], tied_param[1], q) 
                    del p, q
        any_past_block = False

        loaded_block = self.loaded_blocks[model_id]
        if not preload and loaded_block != None:
            any_past_block = True
            self.gpu_unload_blocks(model_id, loaded_block)
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()


        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            # if not preload:
            #     print(f"Request to load model {entry_name} ({model_name}) in GPU")
                

        if self.async_transfers and blocks_name != None:
            first = self.prev_blocks_names[entry_name] == None or not any_past_block
            next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
            if first:
                if self.verboseLevel >=2:
                    if preload:
                        print(f"Preloading model {entry_name} ({model_name}) in GPU")
                    else:
                        print(f"Loading model {entry_name} ({model_name}) in GPU")
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules[entry_name])

            torch.cuda.synchronize()

            if next_blocks_entry != None:
                if self.verboseLevel >=2:
                    print(f"Prefetching model {next_blocks_entry} ({model_name}) in GPU")
                cpu_to_gpu(self.transfer_stream, self.blocks_of_modules[next_blocks_entry]) #, self.default_stream

        else:
            if self.verboseLevel >=2:
                print(f"Loading model {entry_name} ({model_name}) in GPU")
            cpu_to_gpu(self.default_stream, self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()

        if not preload:
            self.loaded_blocks[model_id] = blocks_name           

        # cl.stop()
        # print(f"load time: {cl.format_time_gap()}")

    @torch.compiler.disable()
    def gpu_unload_blocks(self, model_id, blocks_name):
        # cl = clock.start()
        if blocks_name != None:
            self.loaded_blocks[model_id] = None 

        blocks_name = model_id if blocks_name is None else model_id + "/" + blocks_name

        if self.verboseLevel >=2:
            model = self.models[model_id]
            model_name = model._get_name()
            print(f"Unloading model {blocks_name} ({model_name}) from GPU")
 
        blocks_params = self.blocks_of_modules[blocks_name]
        for param in blocks_params:
            parent_module, n, p, is_buffer, _  = param
            if is_buffer:
                q = torch.nn.Buffer(p)
            else:
                q = torch.nn.Parameter(p , requires_grad=False)
            setattr(parent_module, n , q)
            del p, q 
        # cl.stop()
        # print(f"unload time: {cl.format_time_gap()}")

    # @torch.compiler.disable()
    def gpu_load(self, model_id):
        model = self.models[model_id]
        self.active_models.append(model)
        self.active_models_ids.append(model_id)

        self.gpu_load_blocks(model_id, None, True)
        for block_name in self.preloaded_blocks_per_model[model_id]:
            self.gpu_load_blocks(model_id, block_name, True)

    def unload_all(self):
        for model_id in self.active_models_ids:
            self.gpu_unload_blocks(model_id, None)      
            for block_name in self.preloaded_blocks_per_model[model_id]:
                self.gpu_unload_blocks(model_id, block_name)

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
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)
            new_args.append(arg)

        for k in kwargs:
            arg = kwargs[k]
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(torch.bfloat16).cuda(non_blocking=True)             
                elif not arg.is_cuda:
                    arg = arg.cuda(non_blocking=True)             
            new_kwargs[k]= arg
        
        return new_args, new_kwargs

    def ready_to_check_mem(self):
        if self.anyCompiledModule:
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

    def ensure_model_loaded(self, model_id):
        if  model_id in self.active_models_ids:
            return
        # new_model_id = getattr(module, "_mm_id") 
        # do not always unload existing models if it is more efficient to keep in them in the GPU 
        # (e.g: small modules whose calls are text encoders) 
        if not self.can_model_be_cotenant(model_id) :
            self.unload_all()
        self.gpu_load(model_id)

    def hook_preload_blocks_for_compilation(self, target_module, model_id,blocks_name, context):

        # @torch.compiler.disable()
        def preload_blocks_for_compile(module,  *args, **kwargs):
            # some_context = context #for debugging
            if blocks_name != None and blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_load_blocks(model_id, blocks_name)

        # need to be registered before the forward not to be break the efficiency of the compilation chain
        # it should be at the top of the compilation as this type of hook in the middle of a chain seems to break memory performance
        target_module.register_forward_pre_hook(preload_blocks_for_compile)        


    def hook_check_empty_cache_needed(self, target_module, model_id, blocks_name, previous_method,  context):

        qint4quantization =  isinstance(target_module, QModuleMixin) and  target_module.weight!= None and  target_module.weight.qtype == qint4 
        if qint4quantization:
            pass

        if hasattr(target_module, "_mm_id"):
            # no hook for a shared module with no weights (otherwise this will cause models loading / unloading for nothing)
            orig_model_id = getattr(target_module, "_mm_id")
            if self.verboseLevel >=2:
                print(f"Model '{model_id}' shares module '{target_module._get_name()}' with module(s) '{orig_model_id}' ")
            assert not self.any_param_or_buffer(target_module)
            if not isinstance(orig_model_id, list):
                orig_model_id = [orig_model_id]
            orig_model_id.append(model_id)
            setattr(target_module, "_mm_id", orig_model_id)
            target_module.forward = target_module._mm_forward
            return

        def check_empty_cuda_cache(module, *args, **kwargs):
            self.ensure_model_loaded(model_id)
            if blocks_name == None:
                if self.ready_to_check_mem():
                    self.empty_cache_if_needed()
            elif blocks_name != self.loaded_blocks[model_id] and blocks_name not in self.preloaded_blocks_per_model[model_id]:
                self.gpu_load_blocks(model_id, blocks_name)
            if qint4quantization:
                args, kwargs = self.move_args_to_gpu(*args, **kwargs)

            return previous_method(*args, **kwargs) 

        setattr(target_module, "_mm_id", model_id)
        setattr(target_module, "_mm_forward", previous_method)

        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_empty_cuda_cache, target_module), previous_method) )

        
    def hook_change_module(self, target_module, model, model_id, module_id, previous_method):

        def check_change_module(module, *args, **kwargs):            
            self.ensure_model_loaded(model_id)
            # transfer leftovers inputs that were incorrectly created in the RAM (mostly due to some .device tests that returned incorrectly "cpu")
            args, kwargs = self.move_args_to_gpu(*args, **kwargs)
            return previous_method(*args, **kwargs) 
  
        if hasattr(target_module, "_mm_id"):
            return
        setattr(target_module, "_mm_id", model_id)

        setattr(target_module, "forward", functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method) )

        if not self.verboseLevel >=1:
            return

        if module_id == None or module_id =='':
            model_name = model._get_name()
            print(f"Hooked to model '{model_id}' ({model_name})")



    def tune_preloading(self, model_id, current_budget, towers_names):
        preloaded_blocks = {}
        preload_total = 0
        max_blocks_fetch = 0

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks

        if current_budget == 0 or towers_names is None or len(towers_names) == 0 or not self.async_transfers:
            return
        # current_budget = 5000 * ONE_MB
        base_size = self.blocks_of_modules_sizes[model_id] 
        current_budget -= base_size
        if current_budget <= 0:
            if self.verboseLevel >=1:
                print(f"Async loading plan for model '{model_id}' : due to limited budget, beside the async shuttle only only base model ({(base_size)/ONE_MB:0.2f} MB) will be preloaded")
            return
        
        towers = []
        total_size = 0
        for tower_name in towers_names:
            max_floor_size = 0
            tower_size = 0
            floors = []
            prefix = model_id + "/" + tower_name
            for name, size in self.blocks_of_modules_sizes.items():
                if name.startswith(prefix):
                    tower_size += size
                    floor_no = int(  name[len(prefix): ] )
                    floors.append( (name, floor_no, size))
                    max_floor_size = max(max_floor_size, size)

            towers.append( (floors, max_floor_size, tower_size) )
            total_size += tower_size
            current_budget -=  2 * max_floor_size
            if current_budget <= 0:
                if self.verboseLevel >=1:
                    print(f"Async loading plan for model '{model_id}' : due to limited budget, beside the async shuttle only the base model ({(base_size)/ONE_MB:0.2f} MB) will be preloaded")
                return


        for floors, max_floor_size, tower_size in towers:
            tower_budget = tower_size / total_size * current_budget
            preload_blocks_count = int( tower_budget / max_floor_size)
            preload_total += preload_blocks_count * max_floor_size
            max_blocks_fetch = max(max_floor_size, max_blocks_fetch)
            if preload_blocks_count  <= 0:
                if self.verboseLevel >=1:
                    print(f"Async loading plan for model '{model_id}' : due to limited budget, beside the async shuttle only the base model ({(base_size)/ONE_MB:0.2f} MB) will be preloaded")
                return 
            
            nb_blocks= len(floors)
            space_between =  (nb_blocks - preload_blocks_count) / preload_blocks_count 
            cursor = space_between
            first_non_preloaded = None
            prev_non_preloaded = None
            for block in floors:
                name, i, size = block
                if i < cursor:
                    if prev_non_preloaded == None:
                        first_non_preloaded = name
                    else:
                        self.next_blocks_names[prev_non_preloaded] = name
                        self.prev_blocks_names[name] = prev_non_preloaded
                    prev_non_preloaded = name
                else:
                    self.next_blocks_names[name] = None
                    self.prev_blocks_names[name] = None
                    preloaded_blocks[name[ len(model_id) + 1 : ] ] = size
                    cursor += 1 + space_between

            if prev_non_preloaded != None and len(towers) == 1 : 
                self.next_blocks_names[prev_non_preloaded] = first_non_preloaded
                self.prev_blocks_names[first_non_preloaded] = prev_non_preloaded
            else:
                self.next_blocks_names[prev_non_preloaded] = None

        self.preloaded_blocks_per_model[model_id] = preloaded_blocks

        if self.verboseLevel >=1:
            print(f"Async loading plan for model '{model_id}' : {(preload_total+base_size)/ONE_MB:0.2f} MB will be preloaded (base size of {base_size/ONE_MB:0.2f} MB + {preload_total/total_size*100:0.1f}% of recurrent layers data) with a {max_blocks_fetch/ONE_MB:0.2f} MB async" + (" circular" if len(towers) == 1 else "") + " shuttle")


def all(pipe_or_dict_of_modules, pinnedMemory = False, quantizeTransformer = True,  extraModelsToQuantize = None, quantizationType = qint8, budgets= 0, workingVRAM = None, asyncTransfers = True, compile = False, perc_reserved_mem_max = 0, coTenantsMap = None, verboseLevel = -1):
    """Hook to a pipeline or a group of modules in order to reduce their VRAM requirements:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    quantizeTransformer: set True by default will quantize on the fly the video / image model
    pinnedMemory: move models in reserved memor. This allows very fast performance but requires 50% extra RAM (usually >=64 GB)
    extraModelsToQuantize: a list of models to be also quantized on the fly (e.g the text_encoder), useful to reduce bith RAM and VRAM consumption
    budgets: 0 by default (unlimited). If non 0, it corresponds to the maximum size in MB that every model will occupy at any moment
        (in fact the real usage is twice this number). It is very efficient to reduce VRAM consumption but this feature may be very slow
        if pinnedMemory is not enabled
    """
    self = offload()
    self.verboseLevel = verboseLevel
    safetensors2.verboseLevel = verboseLevel
    self.modules_data = {}
    model_budgets = {}

    windows_os =  os.name == 'nt'

    budget = 0
    if not budgets is None:
        if isinstance(budgets , dict):
            model_budgets = budgets
            budget = budgets.get("*", 0) * ONE_MB
        else:
            budget = int(budgets) * ONE_MB

    # if (budgets!= None or budget >0) :
    #     self.async_transfers = True
    self.async_transfers = asyncTransfers



    torch.set_default_device('cpu')

    if hasattr(pipe_or_dict_of_modules, "components"):
        # create a fake Accelerate parameter so that lora loading doesn't change the device
        pipe_or_dict_of_modules.hf_device_map = torch.device("cuda")
        pipe_or_dict_of_modules= pipe_or_dict_of_modules.components 

    
    models = {k: _remove_model_wrapper(v) for k, v in pipe_or_dict_of_modules.items() if isinstance(v, torch.nn.Module)}

    
    verboseLevel = _compute_verbose_level(verboseLevel)

    _welcome()        
    if coTenantsMap != None:
        self.cotenants_map = coTenantsMap 

    self.models = models

    extraModelsToQuantize =  extraModelsToQuantize if extraModelsToQuantize is not None else []
    if not isinstance(extraModelsToQuantize, list):
        extraModelsToQuantize= [extraModelsToQuantize]
    if quantizeTransformer:
        extraModelsToQuantize.append("transformer")            
    models_to_quantize = extraModelsToQuantize

    modelsToPin = []
    pinAllModels = False
    if isinstance(pinnedMemory, bool):
        pinAllModels = pinnedMemory
    elif isinstance(pinnedMemory, list):            
        modelsToPin = pinnedMemory
    else:
        modelsToPin = [pinnedMemory]

    modelsToCompile = []
    compileAllModels = False
    if isinstance(compile, bool):
        compileAllModels = compile
    elif isinstance(compile, list):            
        modelsToCompile = compile
    else:
        modelsToCompile = [compile]

    self.anyCompiledModule = compileAllModels or len(modelsToCompile)>0
    if self.anyCompiledModule:
        torch.compiler.reset()
        torch._dynamo.config.cache_size_limit = 10000
    #dynamic=True

      #  torch._logging.set_logs(recompiles=True)
      #  torch._inductor.config.realize_opcount_threshold = 100 # workaround bug "AssertionError: increase TRITON_MAX_BLOCK['X'] to 4096."

     
    perc_reserved_mem_max = _get_perc_reserved_mem_max(perc_reserved_mem_max) 
    max_reservable_memory = perc_reserved_mem_max * physical_memory

    estimatesBytesToPin = 0
    for model_id in models: 
        current_model: torch.nn.Module = models[model_id] 
        # make sure that no RAM or GPU memory is not allocated for gradiant / training
        current_model.to("cpu").eval()
        
        # if the model has just been quantized so there is no need to quantize it again
        if model_id in models_to_quantize:
            _quantize(current_model, weights=quantizationType, verboseLevel = self.verboseLevel, model_id=model_id)

        modelPinned = (pinAllModels or model_id in modelsToPin) and not hasattr(current_model,"_already_pinned")

        current_model_size = 0

        for n, p in current_model.named_parameters():
            p.requires_grad = False
            if isinstance(p, QTensor):
                # # fix quanto bug (seems to have been fixed)   
                # if not modelPinned and p._scale.dtype == torch.float32:
                #     p._scale = p._scale.to(torch.bfloat16) 
                if p._qtype == qint4:
                    if hasattr(p,"_scale_shift"):
                        current_model_size +=  torch.numel(p._scale_shift) * p._scale_shift.element_size()
                    else:
                        current_model_size +=  torch.numel(p._scale) * p._shift.element_size() + torch.numel(p._scale) * p._shift.element_size()

                    current_model_size +=  torch.numel(p._data._data) * p._data._data.element_size()

                else:
                    current_model_size +=  torch.numel(p._scale) * p._scale.element_size()
                    current_model_size +=  torch.numel(p._data) * p._data.element_size()

            else:
                if p.data.dtype == torch.float32:
                    # convert any left overs float32 weight to bloat16 to divide by 2 the model memory footprint
                    p.data = p.data.to(torch.bfloat16)
                current_model_size +=  torch.numel(p.data) * p.data.element_size()
                            
        for b in current_model.buffers():
            # do not convert 32 bits float to 16 bits since buffers are few (and potential gain low) and usually they are needed for precision calculation (for instance Rope)
            current_model_size +=  torch.numel(b.data) * b.data.element_size()

        if modelPinned:
            estimatesBytesToPin += current_model_size
        

        model_budget = model_budgets[model_id] * ONE_MB if model_id in model_budgets else budget
        if workingVRAM != None:
            model_minimumVRAM = -1
            if isinstance(workingVRAM, dict):
                if model_id in workingVRAM:
                    model_minimumVRAM = workingVRAM[model_id]
                elif "*" in model_id in workingVRAM:
                    model_minimumVRAM = workingVRAM["*"]
            else:
                model_minimumVRAM = workingVRAM
            if model_minimumVRAM > 0:
                new_budget = self.device_mem_capacity -  model_minimumVRAM * ONE_MB
                new_budget = 1 if new_budget  < 0 else new_budget
                model_budget =  new_budget if model_budget == 0 or new_budget < model_budget else model_budget
        if  model_budget > 0 and model_budget > current_model_size:
            model_budget = 0
        coef =0.8
        if current_model_size > coef * self.device_mem_capacity and model_budget == 0 or model_budget > coef * self.device_mem_capacity:
            if verboseLevel >= 1:
                if model_budget == 0:
                    print(f"Model '{model_id}' is too large ({current_model_size/ONE_MB:0.1f} MB) to fit entirely in {coef * 100}% of the VRAM (max capacity is {coef * self.device_mem_capacity/ONE_MB}) MB)")
                else:
                    print(f"Budget ({budget/ONE_MB:0.1f} MB) for Model '{model_id}' is too important so that this model can fit in the VRAM (max capacity is {self.device_mem_capacity/ONE_MB}) MB)")
                print(f"Budget allocation for this model has been consequently reduced to the 80% of max GPU Memory ({coef * self.device_mem_capacity/ONE_MB:0.1f} MB). This may not leave enough working VRAM and you will probably need to define manually a lower budget for this model.")
                model_budget = coef * self.device_mem_capacity 
                
        
        model_budgets[model_id] = model_budget

    partialPinning = False

    if estimatesBytesToPin > 0 and estimatesBytesToPin >= (max_reservable_memory - total_pinned_bytes):
        if self.verboseLevel >=1:
            print(f"Switching to partial pinning since full requirements for pinned models is {estimatesBytesToPin/ONE_MB:0.1f} MB while estimated available reservable RAM is {(max_reservable_memory-total_pinned_bytes)/ONE_MB:0.1f} MB. You may increase the value of parameter 'perc_reserved_mem_max' to a value higher than {perc_reserved_mem_max:0.2f} to force full pinnning." )
        partialPinning = True

    #  Hook forward methods of modules 
    for model_id in models: 
        current_model: torch.nn.Module = models[model_id] 
        towers_names, towers_modules = _detect_main_towers(current_model)
        # compile main iterative modules stacks ("towers")
        compilationInThisOne = compileAllModels or model_id in modelsToCompile 
        if compilationInThisOne:
            if self.verboseLevel>=1:
                if len(towers_modules)>0:
                    formated_tower_names = [name + '*' for name in towers_names]
                    print(f"Pytorch compilation of '{model_id}' is scheduled for these modules : {formated_tower_names}.")
                else:
                    print(f"Pytorch compilation of model '{model_id}' is not yet supported.")

            for submodel in towers_modules:
                submodel.forward= torch.compile(submodel.forward,  backend= "inductor", mode="default" ) # , fullgraph= True, mode= "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",  
                    #dynamic=True,
                
        if pinAllModels or model_id in modelsToPin:
            if hasattr(current_model,"_already_pinned"):
                if self.verboseLevel >=1:
                    print(f"Model '{model_id}' already pinned to reserved memory")
            else:
                _pin_to_memory(current_model, model_id, partialPinning= partialPinning, verboseLevel=verboseLevel)            

        current_budget = model_budgets[model_id]
        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False
        self.loaded_blocks[model_id] = None

        for submodule_name, submodule in current_model.named_modules():  
            # create a fake 'accelerate' parameter so that the _execution_device property returns always "cuda" 
            # (it is queried in many pipelines even if offloading is not properly implemented)  
            if not hasattr(submodule, "_hf_hook"):
                setattr(submodule, "_hf_hook", HfHook())
            if current_budget > 0 and len(submodule_name) > 0:
                if cur_blocks_prefix != None:
                    if submodule_name.startswith(cur_blocks_prefix):
                        depth_prefix = cur_blocks_prefix.split(".")
                        depth_name = submodule_name.split(".")
                        level  =  depth_name[len(depth_prefix)-1]                        
                        pre , num = _extract_num_from_str(level)
                        if num != cur_blocks_seq and not (is_mod_seq and cur_blocks_seq>=0):
                            prev_blocks_name = cur_blocks_name
                            cur_blocks_name =  cur_blocks_prefix + str(num)
                            # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
                        cur_blocks_seq = num
                    else:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_name,cur_blocks_seq, is_mod_seq = None, None, None, -1, False

                if cur_blocks_prefix == None:
                    pre , num = _extract_num_from_str(submodule_name)
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):  
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre + ".", None, -1, isinstance(submodule, torch.nn.Sequential)
                    elif num >=0:
                        cur_blocks_prefix, prev_blocks_name, cur_blocks_seq, is_mod_seq = pre, None, num, False
                        cur_blocks_name = submodule_name
                        # print(f"new block: {model_id}/{cur_blocks_name} - {submodule_name}")
                          
 
            if hasattr(submodule, "forward"):
                submodule_method = getattr(submodule, "forward")
                if callable(submodule_method):   
                    if len(submodule_name.split("."))==1:
                        self.hook_change_module(submodule, current_model, model_id, submodule_name, submodule_method)
                    elif compilationInThisOne and submodule in towers_modules: 
                        self.hook_preload_blocks_for_compilation(submodule, model_id, cur_blocks_name, context = submodule_name )
                    else:
                        self.hook_check_empty_cache_needed(submodule, model_id, cur_blocks_name, submodule_method, context = submodule_name )

                self.add_module_to_blocks(model_id, cur_blocks_name, submodule, prev_blocks_name, submodule_name)

        self.tune_preloading(model_id, current_budget, towers_names)


    if self.verboseLevel >=2:
        start_num, prev_num, prev_pre, prev_size  = -1, -1, None, -1
         
        def print_size_range(n,start_num,prev_num, prev_size ):
            if prev_num < 0:
                print(f"Size of submodel '{n}': {prev_size/ONE_MB:.1f} MB")
            elif prev_num - start_num <=1:
                print(f"Size of submodel '{n+ str(start_num)}': {prev_size/ONE_MB:.1f} MB")
            else:
                print(f"Size of submodel '{n+ str(start_num) +'-'+ str(prev_num)}': {(prev_num-start_num+1)*prev_size/ONE_MB:.1f} MB ({prev_size/ONE_MB:.1f} MB x {prev_num-start_num+1})")

        for n, size in self.blocks_of_modules_sizes.items():
            size = int(size / 10000)* 10000
            pre, num = _extract_num_from_str(n) if "/" in n else (n, -1)
            if prev_pre == None :
                start_num = num
            elif prev_pre != pre or prev_pre == pre and size != prev_size:
                print_size_range(prev_pre,start_num,prev_num, prev_size )
                start_num = num
            prev_num, prev_pre, prev_size = num, pre, size
        if prev_pre != None:
            print_size_range(prev_pre,start_num,prev_num, prev_size )

  
    torch.set_default_device('cuda')
    torch.cuda.empty_cache()
    gc.collect()         

    return self


def profile(pipe_or_dict_of_modules, profile_no: profile_type =  profile_type.VerylowRAM_LowVRAM, verboseLevel = -1, **overrideKwargs):
    """Apply a configuration profile that depends on your hardware:
    pipe_or_dict_of_modules : the pipeline object or a dictionary of modules of the model
    profile_name : num of the profile:
        HighRAM_HighVRAM_Fastest (=1): at least 48 GB of RAM and 24 GB of VRAM : the fastest well suited for a RTX 3090 / RTX 4090
        HighRAM_LowVRAM_Fast (=2): at least 48 GB of RAM and 12 GB of VRAM : a bit slower, better suited for RTX 3070/3080/4070/4080 
            or for RTX 3090 / RTX 4090 with large pictures batches or long videos
        LowRAM_HighVRAM_Medium (=3): at least 32 GB of RAM and 24 GB of VRAM : so so speed but adapted for RTX 3090 / RTX 4090 with limited RAM
        LowRAM_LowVRAM_Slow (=4): at least 32 GB of RAM and 12 GB of VRAM : if have little VRAM or generate longer videos 
        VerylowRAM_LowVRAM_Slowest (=5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work            
    overrideKwargs: every parameter accepted by Offload.All can be added here to override the profile choice
        For instance set quantizeTransformer = False to disable transformer quantization which is by default in every profile
    """      

    _welcome()

    verboseLevel = _compute_verbose_level(verboseLevel)

    modules = pipe_or_dict_of_modules

    if hasattr(modules, "components"):
        modules= modules.components 

    modules = {k: _remove_model_wrapper(v) for k, v in modules.items() if isinstance(v, torch.nn.Module)}
    module_names = {k: _get_module_name(v) for k, v in modules.items() }

    default_extraModelsToQuantize = []
    quantizeTransformer = True
    
    models_to_scan = ("text_encoder", "text_encoder_2")
    candidates_to_quantize = ("t5", "llama", "llm")
    for model_id  in models_to_scan:
        if model_id in module_names: 
            name = module_names[model_id]
            for candidate in candidates_to_quantize:
                if candidate in name:
                    default_extraModelsToQuantize.append(model_id)
                    break


    # transformer (video or image generator) should be as small as possible not to occupy space that could be used by actual image data
    # on the other hand the text encoder should be quite large (as long as it fits in 10 GB of VRAM) to reduce sequence offloading

    budgets = {}
    if "transformer" in modules:
        budgets["transformer"] = 1200    

    extraModelsToQuantize = None
    asyncTransfers = True

    if profile_no == profile_type.HighRAM_HighVRAM:
        pinnedMemory= True
        budgets = None
        info = "You have chosen a profile that may require 48 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.HighRAM_LowVRAM:
        pinnedMemory= True
        budgets["*"] =  3000
        info = "You have chosen a profile that may require 48 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_HighVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets = None
        info = "You have chosen a Medium speed profile that may require 32 GB of RAM and up to 24 GB of VRAM on some applications."
    elif profile_no == profile_type.LowRAM_LowVRAM:
        pinnedMemory= "transformer"
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        info = "You have chosen a profile that usually may require 32 GB of RAM and up to 12 GB of VRAM on some applications."
    elif profile_no == profile_type.VerylowRAM_LowVRAM:
        pinnedMemory= False
        extraModelsToQuantize = default_extraModelsToQuantize
        budgets["*"] =  3000
        if "transformer" in modules:
            budgets["transformer"] = 400    
        #asyncTransfers = False
        info = "You have chosen the slowest profile that may require 24 GB of RAM and up to 10 GB of VRAM on some applications."
    else:
        raise Exception("Unknown profile")
    info += " Actual requirements may varry depending on the application or on the tuning done to the profile."
    
    if budgets != None and len(budgets) == 0:
        budgets = None

    CrLf = '\r\n'
    kwargs = { "pinnedMemory": pinnedMemory,  "extraModelsToQuantize" : extraModelsToQuantize, "budgets": budgets, "asyncTransfers" : asyncTransfers, "quantizeTransformer": quantizeTransformer   }

    if verboseLevel>=2:
        info = info + CrLf + f"Profile '{profile_type.tostr(profile_no)}' sets the following options:" 
        for k,v in kwargs.items():
            if k in overrideKwargs: 
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}' overriden with value '{overrideKwargs[k]}'"
            else:
                info = info + CrLf + f"- '{k}':  '{kwargs[k]}'"

    for k,v in overrideKwargs.items():
        kwargs[k] = overrideKwargs[k]

    if info:
        print(info)

    return all(pipe_or_dict_of_modules, verboseLevel = verboseLevel, **kwargs)
