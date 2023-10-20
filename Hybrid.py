import numpy as np
import torch.nn as nn
from QuantumCircuit import QuantumCircuit
from HybridFunction import HybridFunction

class Hybrid(nn.Module):
    def __init__(self, n_qubits,backend, shots, shift):
        super(Hybrid, self).__init__()
        self.n_qubits = n_qubits
        self.quantum_circuit = QuantumCircuit(self.n_qubits, backend, shots)
        self.shift = shift
        
    def forward(self, input_prm):
        return HybridFunction.apply(input_prm, self.quantum_circuit, self.shift)