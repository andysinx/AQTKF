#qtk.py
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

'''# Pauli matrices
I = np.eye(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

class QuantumTemporalKernel():
    def __init__(self, n_qubits=3, embedding_code="euler", L=20, T=1, features=3, use_noise=False):
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
        scale = 0.5 + np.random.uniform(0, 0.5)
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
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)

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
            self.generate_K_validation_dict(X_train, X_validation)'''





# qtk_cnc.py


# Pauli matrices
I = np.eye(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

class QuantumTemporalKernel:
    def __init__(
        self,
        n_qubits=3,
        L=20,
        T=1,
        features=3,
        n_norm=None,
        n_anom=None,
        optimize=True
    ):
        self.n_qubits = n_qubits
        self.length = L
        self.T = T
        self.N_Features = features

        self.n_norm = n_norm
        self.n_anom = n_anom
        self.optimize = optimize

        # =====================================================
        # K_target NECESSARIO per EVOVAQ
        # =====================================================
        if self.optimize:
            if self.n_norm is None or self.n_anom is None:
                raise ValueError(
                    "n_norm and n_anom must be provided when optimize=True"
                )
            n = self.n_norm + self.n_anom
            self.K_target = np.zeros((n, n))
            self.K_target[:self.n_norm, :self.n_norm] = 1.0
            self.K_target[self.n_norm:, self.n_norm:] = 1.0

        # Hamiltoniano completo
        self.full_terms = []
        self.H = self.generate_full_H()

        # Kernel cache
        self.K_train_dict = {}
        self.K_test_dict = {}
        self.K_validation_dict = {}


    # --------------------------
    # Hamiltoniano
    # --------------------------
    def generate_full_H(self):
        n = self.n_qubits
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)
        labels = ['I', 'X', 'Y', 'Z']

        # Genera tutti i termini tensoriali tranne l'identità
        if not self.full_terms:
            for term in itertools.product(range(4), repeat=n):
                if all(p == 0 for p in term):
                    continue
                label = "".join([labels[p] for p in term])
                self.full_terms.append((label, np.random.uniform(-0.5, 0.5)))

        # Costruisci Hamiltoniano dai coefficienti
        for label, coeff in self.full_terms:
            op = np.array([[1]], dtype=complex)
            for l in label:
                op = np.kron(op, {'I': I, 'X': X, 'Y': Y, 'Z': Z}[l])
            H += coeff * op
        self.H = H
        return H

    # --------------------------
    # Aggiorna solo coefficienti ottimizzabili
    # --------------------------
    def set_H_coeffs(self, coeffs):
        if len(coeffs) != len(self.full_terms):
            raise ValueError("Length of coeffs must match number of full_terms")
        self.full_terms = [(label, c) for (label, _), c in zip(self.full_terms, coeffs)]
        # ricostruisci H
        n = self.n_qubits
        H = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for label, c in self.full_terms:
            op = np.array([[1]], dtype=complex)
            for l in label:
                op = np.kron(op, {'I': I, 'X': X, 'Y': Y, 'Z': Z}[l])
            H += c * op
        self.H = H

    # --------------------------
    # Embedding angle
    # --------------------------
    def embedding_angle(self, x_window, t):
        qc = QuantumCircuit(self.n_qubits)
        t_scaled = 2*np.pi*float(t)/max(1,self.length)*self.T
        U_t = expm(-1j * self.H * t_scaled)
        qc.unitary(Operator(U_t), range(self.n_qubits))
        x_array = np.array(x_window, dtype=float)
        for i in range(len(x_array)):
            qc.rx(np.pi*x_array[i], i)
            qc.rz(np.pi*x_array[i], i)
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)
        return qc

    # --------------------------
    # Fidelity / Similarity
    # --------------------------
    def evaluate_instant_similarity(self, x1, x2, t):
        psi = Statevector(self.embedding_angle(x1, t))
        phi = Statevector(self.embedding_angle(x2, t))
        return float(np.real(abs(np.vdot(psi.data, phi.data))**2))

    # --------------------------
    # Kernel matrix
    # --------------------------
    def generate_K_train(self, X_train):
        n_samples = len(X_train)
        K = np.zeros((n_samples, n_samples), dtype=float)
        for i in tqdm(range(n_samples), desc="Generating K_train"):
            for j in range(i, n_samples):
                sim_sum = 0.0
                for t in range(self.length):
                    sim_sum += self.evaluate_instant_similarity(X_train[i][t], X_train[j][t], t)
                K[i, j] = sim_sum / self.length
                K[j, i] = K[i, j]
            K[i, i] = 1.0
        self.K_train_dict[0] = K
        return K

    def generate_K_test(self, X_train, X_test):
        n_test = len(X_test)
        n_train = len(X_train)
        K = np.zeros((n_test, n_train), dtype=float)
        for i in tqdm(range(n_test), desc="Generating K_test"):
            for j in range(n_train):
                sim_sum = 0.0
                for t in range(self.length):
                    sim_sum += self.evaluate_instant_similarity(X_test[i][t], X_train[j][t], t)
                K[i, j] = sim_sum / self.length
        self.K_test_dict[0] = K
        return K

    def compute_kernels(self, X_train, X_test, X_val=None):
        self.generate_K_train(X_train)
        self.generate_K_test(X_train, X_test)
        if X_val is not None:
            self.generate_K_validation_dict(X_train, X_val)



class QuantumKernel:
    """
    Quantum Kernel hardware-efficient parametrico (R_y + entanglement a cascata)
    Supporta finestre 2D (window_size, n_features) calcolando media fidelity su tutti i timestep.
    """
    def __init__(self, n_qubits=4, feature_order=None, use_noise=False):
        self.n_qubits = n_qubits
        self.use_noise = use_noise
        self.feature_order = feature_order or [f"Feature_{i+1}" for i in range(n_qubits)]
        self.K_train_dict = {}
        self.K_test_dict = {}
        self.K_validation_dict = {}

    # Embedding R_y + CNOT cascade
    def embedding_r_y_cascade(self, x_t):
        x_array = np.array(x_t, dtype=float)
        n_features = len(x_array)
        qc = QuantumCircuit(self.n_qubits)

        qc.h(0)
        for i in range(min(self.n_qubits, n_features)):
            qc.ry(np.pi * x_array[i], i)

        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        for i in range(min(self.n_qubits, n_features)):
            qc.ry(np.pi * x_array[i] / 2, i)

        for i in reversed(range(self.n_qubits - 1)):
            qc.cx(i, i + 1)

        for i in range(min(self.n_qubits, n_features)):
            qc.ry(np.pi * x_array[i] / 3, i)

        return qc

    # Fidelity media tra finestre
    def evaluate_similarity(self, x_window1, x_window2):
        window_len = min(x_window1.shape[0], x_window2.shape[0])
        sim_sum = 0.0
        for t in range(window_len):
            psi = Statevector(self.embedding_r_y_cascade(x_window1[t]))
            phi = Statevector(self.embedding_r_y_cascade(x_window2[t]))
            sim_sum += float(np.real(abs(np.vdot(psi.data, phi.data))**2))
        return sim_sum / window_len

    # Kernel train/test/validation
    def generate_K_train(self, X_train):
        n_samples = len(X_train)
        K = np.zeros((n_samples, n_samples), dtype=float)
        for i in tqdm(range(n_samples), desc="Generating K_train"):
            for j in range(i, n_samples):
                sim = self.evaluate_similarity(X_train[i], X_train[j])
                K[i, j] = sim
                K[j, i] = sim
            K[i, i] = 1.0
        self.K_train_dict[0] = K
        return K

    def generate_K_test(self, X_train, X_test):
        n_test = len(X_test)
        n_train = len(X_train)
        K = np.zeros((n_test, n_train), dtype=float)
        for i in tqdm(range(n_test), desc="Generating K_test"):
            for j in range(n_train):
                K[i, j] = self.evaluate_similarity(X_test[i], X_train[j])
        self.K_test_dict[0] = K
        return K

    def generate_K_validation(self, X_train, X_val):
        n_val = len(X_val)
        n_train = len(X_train)
        K = np.zeros((n_val, n_train), dtype=float)
        for i in tqdm(range(n_val), desc="Generating K_validation"):
            for j in range(n_train):
                K[i, j] = self.evaluate_similarity(X_val[i], X_train[j])
        self.K_validation_dict[0] = K
        return K

    def compute_kernels(self, X_train, X_test, X_val=None):
        self.generate_K_train(X_train)
        self.generate_K_test(X_train, X_test)
        if X_val is not None:
            self.generate_K_validation(X_train, X_val)
