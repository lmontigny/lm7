class <lambda>(torch.nn.Module):
    def forward(self):
        arg4_1: "f32[2, 16]"; 

        arg4_1, = fx_pytree.tree_flatten_spec([], self._in_spec)
        # No stacktrace found for following nodes
        _0_weight: "f32[32, 16]" = getattr(self, "0").weight
        _0_bias: "f32[32]" = getattr(self, "0").bias
        _2_weight: "f32[4, 32]" = getattr(self, "2").weight
        _2_bias: "f32[4]" = getattr(self, "2").bias

        # File: /home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        permute: "f32[16, 32]" = torch.ops.aten.permute.default(_0_weight, [1, 0]);  _0_weight = None
        addmm: "f32[2, 32]" = torch.ops.aten.addmm.default(_0_bias, arg4_1, permute);  _0_bias = arg4_1 = permute = None

        # File: /home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
        relu: "f32[2, 32]" = torch.ops.aten.relu.default(addmm);  addmm = None

        # File: /home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_1: "f32[32, 4]" = torch.ops.aten.permute.default(_2_weight, [1, 0]);  _2_weight = None
        addmm_1: "f32[2, 4]" = torch.ops.aten.addmm.default(_2_bias, relu, permute_1);  _2_bias = relu = permute_1 = None
        return (addmm_1,)
