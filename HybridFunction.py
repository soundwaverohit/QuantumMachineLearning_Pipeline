import torch
import numpy as np
from torch.autograd import Function



class HybridFunction(Function):    
    @staticmethod
    def forward(ctx, input_prm, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        
        expectation_z = ctx.quantum_circuit.run(input_prm[0].tolist())
        result = torch.tensor([expectation_z])
    
        ctx.save_for_backward(input_prm, result)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        
        input_prm, expectation_z = ctx.saved_tensors
        input_list = np.array(input_prm.tolist())
        
        shift_right = input_list + np.ones(input_list.shape)*ctx.shift
        shift_left  = input_list - np.ones(input_list.shape)*ctx.shift
        
        gradients = []
        
        for i in range(len(input_list)):
            expec_right = ctx.quantum_circuit.run(shift_right[i])
            expec_left  = ctx.quantum_circuit.run(shift_left[i])
            
            gradient = torch.tensor([expec_right]) - torch.tensor([expec_left])
            gradients.append(gradient)
            
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None
        
    
    