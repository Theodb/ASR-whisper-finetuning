import torch

def count_parameters(model):
    # Define column headers with appropriate spacing
    header_modules = "Modules"
    header_trainable = "Trainable Parameters"
    header_non_trainable = "Non-trainable Parameters"
    
    # Define the width for each column
    width_modules = 40
    width_trainable = 25
    width_non_trainable = 25
    
    # Print the table header
    print(f"\n{header_modules:<{width_modules}} {header_trainable:>{width_trainable}} {header_non_trainable:>{width_non_trainable}}")
    print("-" * (width_modules + width_trainable + width_non_trainable + 2))  # Separator line
    
    total_trainable = 0  # Initialize total trainable parameters
    
    # Iterate over each child module in the model
    for child in model.children():
        trainable_group_params = 0  # Trainable parameters in this group
        non_trainable_group_params = 0  # Non-trainable parameters in this group
        group_name = type(child).__name__  # Name of the module/group
        
        # Iterate over all parameters in the child module
        for name, parameter in child.named_parameters():
            if parameter.requires_grad:
                trainable_group_params += parameter.numel()
            else:
                non_trainable_group_params += parameter.numel()
        
        # Print the parameters for this module
        print(f"{group_name:<{width_modules}} {trainable_group_params:>{width_trainable},} {non_trainable_group_params:>{width_non_trainable},}")
        
        # Accumulate the total trainable parameters
        total_trainable += trainable_group_params
    
    # Print the separator line after the table
    print("-" * (width_modules + width_trainable + width_non_trainable + 2))
    
    # Print the total number of trainable parameters
    print(f"{'Total Trainable Params:':<{width_modules}} {total_trainable:>{width_trainable},}")
    
    return total_trainable

# ################ monkey patch for quanto
def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
      name, val = named_parameter
      flag = True
      if hasattr(val,"_data") or hasattr(val,"_scale"):
        if hasattr(val,"_data"):
          yield name + "._data", val._data
        if hasattr(val,"_scale"):
          yield name + "._scale", val._scale
      else:
        yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
      yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def compute_module_sizes(model):
    """
    Compute the size of each submodule of a given model.
    """
    from collections import defaultdict
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
      size = tensor.numel() * dtype_byte_size(tensor.dtype)
      name_parts = name.split(".")
      for idx in range(len(name_parts) + 1):
        module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes