# qtk.py
"""
QuantumTemporalKernel: Temporal Quantum Kernel based on parametric Hamiltonian evolution.
Supports:
 - local terms λ_i * Z_i
 - static couplings mu_pairs -> Z_i Z_j
 - dynamic couplings nu_pairs -> (X_i X_j + Y_i Y_j)
 - embedding based on:
     * angle encoding RX/RZ (default)
     * efficient_su2 (flag embedding_type="efficient_su2")
"""

import re
import itertools
import numpy as np
from tqdm import tqdm
from scipy.linalg import expm
from qiskit.circuit.library import Initialize

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import ParameterVector

# Sampler import in try/except per compatibilità ambienti
try:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer.noise import NoiseModel
except Exception:
    Sampler = None
    FakeTorino = None
    NoiseModel = None

# Pauli matrices
I = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


class QuantumTemporalKernel:
    def __init__(
        self,
        n_qubits=8,
        embedding_type="angle",   # "angle" o "efficient_su2"
        L=20,
        T=1,
        features=8,
        use_noise=False,
        mu_pairs=None,
        nu_pairs=None,
        corr_pairs=None,
        alpha=0.25,
        beta=0.10,
        feature_order=None,
        lambda_z=None,
        n_norm=None, 
        n_anom=None, 
        optimize=True
    ):
        self.n_qubits = int(n_qubits)
        self.embedding_type = embedding_type
        self.length = int(L)
        self.T = T
        self.N_Features = int(features)
        self.use_noise = bool(use_noise)
        self.n_norm = n_norm
        self.n_anom = n_anom
        self.optimize = optimize
        if self.optimize:
            n = self.n_norm + self.n_anom
            self.K_target = np.zeros((n, n))
            self.K_target[:self.n_norm, :self.n_norm] = 1
            self.K_target[self.n_norm:, self.n_norm:] = 1

        # mappatura qubit -> feature name
        self.feature_order = feature_order or [
            "Accelerometer1RMS", "Accelerometer2RMS", "Current",
            "Pressure", "Temperature", "Thermocouple",
            "Voltage", "Volume Flow RateRMS"
        ]
        if len(self.feature_order) != self.n_qubits:
            fo = list(self.feature_order)[:self.n_qubits]
            while len(fo) < self.n_qubits:
                fo.append(f"Feature_{len(fo)+1}")
            self.feature_order = fo

        # Simulator
        self.simulator = self._setup_simulator()
        self.sampler = None
        if Sampler is not None:
            try:
                self.sampler = Sampler(mode=self.simulator)
            except Exception:
                self.sampler = None

        # Accoppiamenti
        if mu_pairs is not None and nu_pairs is not None:
            self.mu_pairs = dict(mu_pairs)
            self.nu_pairs = dict(nu_pairs)
        elif corr_pairs is not None:
            self.mu_pairs, self.nu_pairs = self._generate_from_corr(corr_pairs, alpha=alpha, beta=beta)
        else:
            keys = [f"{i+1}-{j+1}" for i in range(self.n_qubits) for j in range(i+1, self.n_qubits)]
            self.mu_pairs = {k: float(np.round(np.random.uniform(-0.05, 0.05), 6)) for k in keys}
            self.nu_pairs = {k: float(np.round(np.random.uniform(-0.02, 0.02), 6)) for k in keys}

        # Lambda locali
        if lambda_z is not None:
            self.lambda_z = np.array(lambda_z, dtype=float)
            if len(self.lambda_z) != self.n_qubits:
                raise ValueError("lambda_z length must equal n_qubits")
        else:
            self.lambda_z = self._default_lambda_for_features()

        # Hamiltoniano
        self.full_terms = []
        self.H = self.generate_full_H()

        # Dizionari kernel
        self.K_train_dict = {}
        self.K_test_dict = {}
        self.K_validation_dict = {}

    # --------------------------
    # Simulator / sampler
    # --------------------------
    def _setup_simulator(self):
        if self.use_noise and FakeTorino is not None and NoiseModel is not None:
            try:
                backend_fake = FakeTorino()
                noise_model = NoiseModel.from_backend(backend_fake)
                return AerSimulator(noise_model=noise_model, method="density_matrix")
            except Exception:
                return AerSimulator(method="statevector")
        else:
            return AerSimulator(method="statevector")

    # --------------------------
    # default Lambda
    # --------------------------
    def _default_lambda_for_features(self):
        lambdas = np.zeros(self.n_qubits, dtype=float)
        for idx, fname in enumerate(self.feature_order):
            name = fname.lower()
            if "accelerometer" in name or "accel" in name:
                lambdas[idx] = 0.30
            elif "current" in name or "voltage" in name:
                lambdas[idx] = 0.12
            elif "pressure" in name or "flow" in name or "rate" in name:
                lambdas[idx] = 0.15
            elif "temp" in name or "thermo" in name:
                lambdas[idx] = 0.07
            else:
                lambdas[idx] = 0.10
        return lambdas

    # --------------------------
    # Correlation for Mu/Nu
    # --------------------------
    def _generate_from_corr(self, corr_pairs, alpha=0.25, beta=0.10, threshold=0.05):
        mu_pairs = {}
        nu_pairs = {}
        for pair, val in corr_pairs.items():
            try:
                if val is None or np.isnan(val): continue
                if abs(val) < threshold: continue
                mu_pairs[pair] = float(np.round(alpha * val, 6))
                nu_pairs[pair] = float(np.round(beta * val, 6))
            except Exception:
                continue
        return mu_pairs, nu_pairs

    # --------------------------
    # Hamiltonian
    # --------------------------
    def generate_H(self):
        n = self.n_qubits
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)

        # Termini locali
        for i in range(n):
            H += self.lambda_z[i] * self._pauli_op(n, [(i, Z)])

        # Termini statici Z_i Z_j
        for pair, coeff in self.mu_pairs.items():
            i, j = self._parse_pair(pair)
            H += coeff * self._pauli_op(n, [(i, Z), (j, Z)])

        # Termini dinamici X_i X_j + Y_i Y_j
        for pair, coeff in self.nu_pairs.items():
            i, j = self._parse_pair(pair)
            H += coeff * (self._pauli_op(n, [(i, X), (j, X)]) + self._pauli_op(n, [(i, Y), (j, Y)]))

        self.H = H
        return H
    
    def generate_full_H(self, coeffs=None):
            """
            Genera un Hamiltoniano completo su n_qubits come combinazione lineare
            di tutti i prodotti tensoriali di matrici di Pauli {I, X, Y, Z}.
            Se coeffs è None, genera coefficienti casuali, altrimenti usa quelli passati.
            """
            n = self.n_qubits
            dim = 2 ** n
            H = np.zeros((dim, dim), dtype=complex)

            labels = ['I', 'X', 'Y', 'Z']

            # genera full_terms solo se vuoto (prima volta)
            if not self.full_terms:
                for term in itertools.product(range(4), repeat=n):
                    if all(p == 0 for p in term):
                        continue
                    label = "".join([labels[p] for p in term])
                    self.full_terms.append((label, 0.0))

            # se coeffs non forniti, genera casuali
            if coeffs is None:
                coeffs = [np.random.uniform(-0.5, 0.5) for _ in self.full_terms]

            if len(coeffs) != len(self.full_terms):
                raise ValueError("Length of coeffs must match number of full_terms")

            for (label,_), c in zip(self.full_terms, coeffs):
                op = np.array([[1]], dtype=complex)
                for l in label:
                    op = np.kron(op, {'I':I,'X':X,'Y':Y,'Z':Z}[l])
                H += c * op

            # aggiorna full_terms con i nuovi coefficienti
            self.full_terms = [(label,c) for (label,_), c in zip(self.full_terms, coeffs)]
            self.H = H
            return H
    
    def set_H_coeffs(self, coeffs):
        """
        Imposta i coefficienti dell'Hamiltoniano completo generato da generate_full_H.
        """
        if not self.full_terms:
            self.generate_full_H()
        self.generate_full_H(coeffs=coeffs)

    # --------------------------
    # Parsing pair
    # --------------------------
    def _parse_pair(self, pair_key):
        if isinstance(pair_key, (tuple, list)) and len(pair_key) >= 2:
            a, b = int(pair_key[0]), int(pair_key[1])
            if a == 0 or b == 0: return a, b
            return a-1, b-1
        s = str(pair_key).strip()
        nums = re.findall(r'\d+', s)
        if len(nums) >= 2:
            return int(nums[0])-1, int(nums[1])-1
        if s.isdigit() and len(s)==2:
            return int(s[0])-1, int(s[1])-1
        raise ValueError(f"Invalid pair key: {pair_key}")

    # --------------------------
    # Pauli tensored operator
    # --------------------------
    def _pauli_op(self, n, ops):
        mat = np.array([[1]], dtype=complex)
        for q in range(n):
            found = False
            for (idx, pmat) in ops:
                if idx==q:
                    mat = np.kron(mat, pmat)
                    found = True
                    break
            if not found:
                mat = np.kron(mat, I)
        return mat

    # ==========================
    # Embedding (switchable)
    # ==========================
    def embedding(self, x_window, t):
        if self.embedding_type == "efficient_su2":
            return self.embedding_efficient_su2(x_window, reps=3)
        elif self.embedding_type == "amplitude":
            # usa use_mean=True per mantenere comportamento simile all'angle (media temporale)
            return self.embedding_amplitude(x_window, t, use_mean=True)
        else:
            return self.embedding_angle(x_window, t)
        

    # ===========================
    # Amplitude embedding (nuova)
    # ===========================
    def embedding_amplitude(self, x_window, t, use_mean=True):
        """
        Amplitude embedding su n_qubits:
        - x_window: array-like. Se 2D (time x features) default prende la media temporale.
                    Se vuoi tutte le entry (flatten temporale) imposta use_mean=False.
        - t: tempo istante (usato per l'evoluzione Hamiltoniana dopo la preparazione di stato).
        Restituisce un QuantumCircuit che prepara lo stato amplitude-encoded e poi applica U(t).
        """
        n = self.n_qubits
        dim = 2 ** n

        # Estrai vettore di feature dall'input
        x_array = np.array(x_window, dtype=float)
        if x_array.ndim == 2:
            if use_mean:
                vec = np.mean(x_array, axis=0)       # media sulle osservazioni temporali (shape = n_features)
            else:
                vec = np.ravel(x_array)              # usa tutte le entries (flatten temporale)
        elif x_array.ndim == 1:
            vec = x_array
        elif x_array.ndim == 0:
            vec = np.array([float(x_array)])
        else:
            vec = np.ravel(x_array)

        # Se ci sono più elementi di dim, trunca; se meno, pad con zeri
        if len(vec) > dim:
            vec = vec[:dim]
        elif len(vec) < dim:
            pad = np.zeros(dim - len(vec), dtype=float)
            vec = np.concatenate([vec, pad])

        # Normalizza L2: evitare divisione per 0
        norm = np.linalg.norm(vec)
        if norm == 0:
            # se il vettore è tutto zero, crea uno stato |0...0>
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 + 0.0j
        else:
            state = np.array(vec, dtype=complex) / norm

        # Costruisci il circuito che inizializza lo stato alle ampiezze date
        qc = QuantumCircuit(n)
        # Initialize richiede vettore di dimensione 2^n
        init = Initialize(state)
        qc.append(init, list(range(n)))
        qc.barrier()

        # Applica evoluzione Hamiltoniana come negli altri embedding
        t_scaled = 2 * np.pi * (t + 1) / self.length * self.T
        U_t = expm(-1j * self.H * t_scaled)
        qc.unitary(Operator(U_t), range(n))

        return qc

    # --------------------------
    # Angle encoding originale
    # --------------------------
    def embedding_angle(self, x_window, t):
        qc = QuantumCircuit(self.n_qubits)
        x_array = np.array(x_window, dtype=float)
        if x_array.ndim == 2:
            avg_features = np.mean(x_array, axis=0)
        elif x_array.ndim ==1:
            avg_features = x_array
        elif x_array.ndim==0:
            avg_features = np.repeat(float(x_array), self.n_qubits)
        else:
            raise ValueError(f"x_window shape unexpected: {x_array.shape}")
        avg_features = np.clip(avg_features, -1.0, 1.0)
        for i in range(self.n_qubits):
            v = float(avg_features[i % len(avg_features)])
            qc.rx(v*np.pi, i)
            qc.rz(v*np.pi, i)
        # entangling
        for i in range(self.n_qubits-1):
            qc.cz(i, i+1)
        if self.n_qubits>1:
            qc.cx(self.n_qubits-1,0)
        # Hamiltoniano
        t_scaled = 2*np.pi*float(t)/max(1,self.length)*self.T
        U_t = expm(-1j*self.H*t_scaled)
        qc.unitary(Operator(U_t), range(self.n_qubits))
        return qc

    # --------------------------
    # Efficient SU2 embedding
    # --------------------------
    def embedding_efficient_su2(self, x_window, n_qubits_embed=3, reps=1):
        qc = QuantumCircuit(n_qubits_embed)

        # Flatten / media temporale
        x_array = np.array(x_window, dtype=float)
        if x_array.ndim == 2:
            x_flat = np.mean(x_array, axis=0)
        elif x_array.ndim == 1:
            x_flat = x_array
        else:
            x_flat = np.ravel(x_array)
        x_flat = np.clip(x_flat, -1.0, 1.0)

        # Numero di parametri EfficientSU2
        su2_template = EfficientSU2(n_qubits_embed, reps=reps)
        n_params = su2_template.num_parameters

        # Adatta features ai parametri
        features_for_su2 = np.tile(x_flat, int(np.ceil(n_params / len(x_flat))))[:n_params]

        # Crea ParameterVector temporaneo
        params = ParameterVector("θ", n_params)

        # Applica EfficientSU2 parametrico
        su2_template = EfficientSU2(n_qubits_embed, reps=reps, entanglement='full')
        qc.compose(su2_template.assign_parameters(dict(zip(su2_template.parameters, features_for_su2))), inplace=True)

        # Evoluzione Hamiltoniana
        t_scaled = 2 * np.pi * float(self.T) / max(1, self.length) * self.T
        H_embed = self.H[:2 ** n_qubits_embed, :2 ** n_qubits_embed]
        U_t = expm(-1j * H_embed * t_scaled)
        qc.unitary(Operator(U_t), range(n_qubits_embed))

        return qc

    # --------------------------
    # Fidelity / Similarity
    # --------------------------
    def evaluate_instant_similarity(self, x1, x2, t):
        psi = Statevector(self.embedding(x1, t))
        phi = Statevector(self.embedding(x2, t))
        return float(np.real(abs(np.vdot(psi.data, phi.data))**2))

    def evaluate_instant_similarity_sampler(self, x1, x2, t, shots=1024):
        if self.sampler is None:
            raise RuntimeError("Sampler not available")
        qc1 = self.embedding(x1, t)
        qc2 = self.embedding(x2, t)
        qc = qc1.compose(qc2.inverse())
        cr = ClassicalRegister(self.n_qubits, name="cr")
        qc.add_register(cr)
        qc.measure(range(self.n_qubits), cr)
        job = self.sampler.run([qc], shots=shots)
        result = job.result()
        counts = {}
        try:
            counts = result[0].data.cr.get_counts()
        except Exception:
            try:
                counts = result.get_counts(0)
            except Exception:
                pass
        p0 = counts.get('0'*self.n_qubits, 0)
        return float(p0)/shots

    # --------------------------
    # Kernel generation
    # --------------------------
    def generate_K_train(self, X_train):
        n_samples = len(X_train)
        combs = list(itertools.combinations(range(n_samples), 2))
        K_avg = np.zeros((n_samples,n_samples),dtype=float)
        for t in tqdm(range(self.length), desc="Generating K_train"):
            for i,j in combs:
                xi = X_train[i][t]
                xj = X_train[j][t]
                s = self.evaluate_instant_similarity(xi,xj,t)
                K_avg[i,j] += s
                K_avg[j,i] += s
            for i in range(n_samples):
                K_avg[i,i] += 1.0
        K_avg /= float(self.length)
        K_avg = 0.5*(K_avg+K_avg.T)
        self.K_train_dict[0] = K_avg
        return K_avg

    def generate_K_test(self, X_train, X_test):
        n_test = len(X_test)
        n_train = len(X_train)
        pairs = list(itertools.product(range(n_test), range(n_train)))
        K_avg = np.zeros((n_test,n_train),dtype=float)
        for t in tqdm(range(self.length), desc="Generating K_test"):
            for i,j in pairs:
                xi = X_test[i][t]
                yj = X_train[j][t]
                K_avg[i,j] += self.evaluate_instant_similarity(xi,yj,t)
        K_avg /= float(self.length)
        self.K_test_dict[0] = K_avg
        return K_avg

    def generate_K_validation_dict(self, X_train, X_validation):
        n_val = len(X_validation)
        n_train = len(X_train)
        pairs = list(itertools.product(range(n_val), range(n_train)))
        K_avg = np.zeros((n_val,n_train),dtype=float)
        for t in tqdm(range(self.length), desc="Generating K_validation"):
            for i,j in pairs:
                xi = X_validation[i][t]
                yj = X_train[j][t]
                K_avg[i,j] += self.evaluate_instant_similarity(xi,yj,t)
        K_avg /= float(self.length)
        self.K_validation_dict[0] = K_avg
        return K_avg

    def compute_kernels(self, X_train, X_test, X_val=None):
        self.generate_K_train(X_train)
        self.generate_K_test(X_train,X_test)
        if X_val is not None:
            self.generate_K_validation_dict(X_train,X_val)
