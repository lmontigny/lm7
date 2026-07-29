


def forward(self, p_0_weight, p_0_bias, p_2_weight, p_2_bias, input):
    input_1 = input
    linear = torch.ops.aten.linear.default(input_1, p_0_weight, p_0_bias);  input_1 = p_0_weight = p_0_bias = None
    relu = torch.ops.aten.relu.default(linear);  linear = None
    linear_1 = torch.ops.aten.linear.default(relu, p_2_weight, p_2_bias);  relu = p_2_weight = p_2_bias = None
    return (linear_1,)
    