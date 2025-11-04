import itertools
from tqdm import tqdm
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
from scipy.linalg import expm
import numpy as np

# Pauli matrices
I = np.eye(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

class QuantumTemporalKernel():
    def __init__(self, n_qubits=3, embedding_code="euler", L=100, T=1, features=3, use_noise=False):
        self.n_qubits = n_qubits
        self.embedding_code = embedding_code
        self.use_noise = use_noise
        self.simulator = self._setup_simulator()
        self.sampler = Sampler(mode=self.simulator)
        self.length = L
        self.T = T
        self.N_Features = features

        # Genera Hamiltoniano casuale con lambda random e mu fissi
        self.lambda_i = np.random.uniform(-2, 2, self.n_qubits)
        self.mu_ij = {'XY': 0.2, 'XZ': 0.2, 'YZ': 0.2}
        self.H = self.generate_H()

        # dizionari per kernel
        self.K_train_dict = {}
        self.K_test_dict = {}
        self.K_validation_dict = {}

    def _setup_simulator(self):
        if self.use_noise:
            backend_fake = FakeTorino()
            noise_model = NoiseModel.from_backend(backend_fake)
            simulator = AerSimulator(noise_model=noise_model, method="density_matrix", device="GPU")
            simulator.set_options(precision='single')
        else:
            simulator = AerSimulator(method="statevector", device="GPU")
        return simulator

    def generate_H(self):
        """Genera un Hamiltoniano non banale con termini X, Z e accoppiamenti XX, ZZ."""
        # coefficienti casuali per i termini singoli
        lambda_z = np.random.uniform(-0.5, 0.5, self.n_qubits)
        lambda_x = np.random.uniform(-0.5, 0.5, self.n_qubits)

        # accoppiamenti fissi o deboli
        scale = 0.1
        mu_z = {k: v*scale for k,v in {'XY': -0.114, 'XZ': 0.067, 'YZ': -0.061}.items()}
        mu_xx = {k: v*scale for k,v in {'XY': -0.127, 'XZ': 0.075, 'YZ': -0.066}.items()}
        mu_yy = {k: v*scale for k,v in {'XY': -0.150, 'XZ': 0.070, 'YZ': -0.090}.items()}

        # operatori singoli su 3 qubit
        Z1 = np.kron(np.kron(Z, I), I)
        Z2 = np.kron(np.kron(I, Z), I)
        Z3 = np.kron(np.kron(I, I), Z)
        X1 = np.kron(np.kron(X, I), I)
        X2 = np.kron(np.kron(I, X), I)
        X3 = np.kron(np.kron(I, I), X)

        # accoppiamenti
        Z1Z2 = np.kron(np.kron(Z, Z), I)
        Z1Z3 = np.kron(np.kron(Z, I), Z)
        Z2Z3 = np.kron(np.kron(I, Z), Z)
        X1X2 = np.kron(np.kron(X, X), I)
        X1X3 = np.kron(np.kron(X, I), X)
        X2X3 = np.kron(np.kron(I, X), X)
        Y1Y2 = np.kron(np.kron(Y, Y), I)
        Y1Y3 = np.kron(np.kron(Y, I), Y)
        Y2Y3 = np.kron(np.kron(I, Y), Y)
        

        # Hamiltoniano totale
        H = (
            lambda_z[0] * Z1 + lambda_z[1] * Z2 + lambda_z[2] * Z3 +
            lambda_x[0] * X1 + lambda_x[1] * X2 + lambda_x[2] * X3 +
            mu_z['XY'] * Z1Z2 + mu_z['XZ'] * Z1Z3 + mu_z['YZ'] * Z2Z3 + 
            mu_xx['XY'] * X1X2 + mu_xx['XZ'] * X1X3 + mu_xx['YZ'] * X2X3 +
            mu_yy['XY'] * Y1Y2 + mu_yy['XZ'] * Y1Y3 + mu_yy['YZ'] * Y2Y3
        )

        self.H = H
        return H

    def embedding(self, x_window, t):
        qc = QuantumCircuit(self.n_qubits)

        # se x_window è scalare → lo trasformiamo in vettore ripetuto
        if np.ndim(x_window) == 1:
            avg_features = x_window
        else:
            avg_features = np.mean(x_window, axis=0)

        # Se per qualche motivo è uno scalare, replicalo su tutti i qubit
        if np.isscalar(avg_features):
            avg_features = np.repeat(avg_features, self.n_qubits)
            
        for i in range(self.n_qubits):
            f = np.clip(avg_features[i % len(avg_features)], -1, 1)
            qc.rx(f * np.pi, i)
            qc.rz(f * np.pi, i)

        # evoluzione temporale
        U_t = expm(-1j * self.H * 2 * np.pi * t / self.length * self.T)
        qc.unitary(Operator(U_t), range(self.n_qubits))
        return qc

    def evaluate_instant_similarity(self, x1, x2, t):
        psi = Statevector(self.embedding(x1, t))
        phi = Statevector(self.embedding(x2, t))
        inner_product = abs(np.vdot(psi.data, phi.data)) ** 2
        return np.real(inner_product)
    
    def evaluate_instant_similarity_sampler(self, x1, x2, t, shots=1024):
        # Generate embeddings for x1 and x2
        qc1 = self.embedding(x1, t)
        qc2 = self.embedding(x2, t)
        
        # Compose the circuits: qc1 followed by inverse of qc2
        qc = qc1.compose(qc2.inverse())

        # Add classical register
        cr = ClassicalRegister(self.n_qubits, name='cr')
        qc.add_register(cr)
        
        # Measure all qubits into the classical register
        qc.measure(range(self.n_qubits), cr)
        
        # Execute on Sampler (probabilistic)
        job = self.sampler.run([qc], shots=shots)
        result = job.result()
        
        # Get counts from the named classical register
        counts = result[0].data.cr.get_counts()  

        # Estimated fidelity = probability of measuring |0...0>
        p0 = counts.get('0'*self.n_qubits, 0)
        return p0 / shots


    def generate_K_train(self, X_train):
        n_samples = X_train.shape[0]
        train_combinations = list(itertools.combinations(range(n_samples), 2))
        K_avg = np.zeros((n_samples, n_samples), dtype=float)

        for t in tqdm(range(self.length)):
            for i, j in train_combinations:
                x = X_train[i][t]
                y = X_train[j][t]
                s = self.evaluate_instant_similarity(x, y, t)   # contributo corrente
                K_avg[i, j] += s
                K_avg[j, i] += s

            # diagonale: aggiungi 1 per ciascun sample all'istante t
            for i in range(n_samples):
                K_avg[i, i] += 1.0

        # media temporale
        K_avg /= float(self.length)
        # sicurezza: forza simmetria numerica residua
        K_avg = 0.5 * (K_avg + K_avg.T)
        self.K_train_dict[0] = K_avg
        print("Averaged K_train_dict generated")


    def generate_K_test(self, X_train, X_test):
        n_test = X_test.shape[0]
        n_train = X_train.shape[0]
        test_combinations = list(itertools.product(range(n_test), range(n_train)))
        K_avg = np.zeros((n_test, n_train))

        for t in tqdm(range(self.length)):
            for i, j in test_combinations:
                x = X_test[i][t]
                y = X_train[j][t]
                K_avg[i, j] += self.evaluate_instant_similarity(x, y, t)

        K_avg /= self.length
        self.K_test_dict[0] = K_avg
        print("Averaged K_test_dict generated")

    def generate_K_validation_dict(self, X_train, X_validation):
        n_validation = X_validation.shape[0]
        n_train = X_train.shape[0]
        validation_combinations = list(itertools.product(range(n_validation), range(n_train)))
        K_avg = np.zeros((n_validation, n_train))

        for t in tqdm(range(self.length)):
            for i, j in validation_combinations:
                x = X_validation[i][t]
                y = X_train[j][t]
                K_avg[i, j] += self.evaluate_instant_similarity(x, y, t)

        K_avg /= self.length
        self.K_validation_dict[0] = K_avg
        print("Averaged K_validation_dict generated")

    def compute_kernels(self, X_train, X_test, X_validation=None):
        self.generate_K_train(X_train)
        self.generate_K_test(X_train, X_test)
        if X_validation is not None:
            self.generate_K_validation_dict(X_train, X_validation)