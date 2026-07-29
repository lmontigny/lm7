
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_bernard'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.replay_side_effects = True
torch._dynamo.config.side_effect_replay_policy = 'info'
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.allow_rnn = False
torch._inductor.config.cpp_wrapper = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.autotune_cublasLt = False
torch._inductor.config.triton.autotune_at_compile_time = True
torch._inductor.config.triton.store_cubin = True
torch._inductor.config.aot_inductor.output_path = 'cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6'
torch._inductor.config.aot_inductor.serialized_in_spec = '[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}]'
torch._inductor.config.aot_inductor.serialized_out_spec = '[1, {"type": null, "context": null, "children_spec": []}]'
torch._inductor.config.aot_inductor.package = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.debug_dir = '/home/bernard/dev/lm7/.model-debug.lm7-14ur7g_p/debug'
torch._inductor.config.trace.fx_graph = True
torch._inductor.config.trace.fx_graph_transformed = True
torch._inductor.config.trace.ir_pre_fusion = True
torch._inductor.config.trace.ir_post_fusion = True
torch._inductor.config.trace.output_code = True
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.13.0+cu130
# torch cuda version: 13.0
# torch git version: cf30153c4c131c8164ee7798e5022d810682e2cb


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 4070 SUPER : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.0 = Module()
        self.2 = Module()



    def forward(self):
        arg4_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        _0_weight = getattr(self, "0").weight
        _0_bias = getattr(self, "0").bias
        _2_weight = getattr(self, "2").weight
        _2_bias = getattr(self, "2").bias
        permute = torch.ops.aten.permute.default(_0_weight, [1, 0]);  _0_weight = None
        addmm = torch.ops.aten.addmm.default(_0_bias, arg4_1, permute);  _0_bias = arg4_1 = permute = None
        relu = torch.ops.aten.relu.default(addmm);  addmm = None
        permute_1 = torch.ops.aten.permute.default(_2_weight, [1, 0]);  _2_weight = None
        addmm_1 = torch.ops.aten.addmm.default(_2_bias, relu, permute_1);  _2_bias = relu = permute_1 = None
        return (addmm_1,)

def load_args(reader):
    buf0 = reader.storage(None, 128)
    reader.tensor(buf0, (2, 16), is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)