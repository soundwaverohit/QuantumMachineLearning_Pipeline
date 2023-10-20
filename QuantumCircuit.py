import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import * 
import numpy as np


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        all_qubits = [i for i in range(n_qubits)]
        
        # Unleash your creativity here - use different circuits and schemes. Go bananas !
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        
        self.backend = backend
        self.shots = shots
        
    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        q_obj = assemble(t_qc, 
                         shots = self.shots,
                         parameter_binds = [{self.theta : theta} for theta in thetas])
        
        job = self.backend.run(q_obj)
        result = job.result().get_counts()
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        probs = counts/self.shots
        
        expec = np.sum(states*probs)
        
        return np.array([expec])