import numpy as np
import torch
import torchvision
import numpy as np
#TODO：导入自定义连接库
import hsigmoid_cpp

def hsigmoid_cpu(rand):
    rand = rand.contiguous()
    #TODO：调用hsigmoid函数对rand进行处理得到输出结果output
    output = hsigmoid_cpp.hsigmoid_cpu(rand)
    return output.contiguous()

def test_hsigmoid():
    torch.manual_seed(12345)
    rand = (torch.randn(3, 512, 512, dtype=torch.float32).abs()+1)
    #TODO：调用hsigmoid_cpu函数对rand进行处理得到输出结果output_cpu
    output_cpu = hsigmoid_cpu(rand)
    print("------------------hsigmoid test completed----------------------")
    print("input: ", rand)
    print("input_size:", rand.size())
    print("output: ", output_cpu)
    print("output_size:", output_cpu.size())

    print("TEST hsigmoid PASS!\n")
    
test_hsigmoid()
