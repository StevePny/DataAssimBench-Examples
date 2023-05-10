"""Class for reservoir computing model for NWP"""

import warnings
import logging
from copy import deepcopy

from scipy import sparse, stats, linalg
import numpy as np
from dabench import data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)


class RCModel():
    """Class for a simple Reservoir Computing data-driven model

    Attributes:
        system_dim (int): dimension of the underlying dynamical system
        time_dim (int): the dimension of the timeseries
        sparsity (float): the percentage of zero-valued entries in A.
            Default: 0.99
        reservoir_dim (int): 
        input_dim (int):
        sigma_bias: 1
        leak_rate,


        A (array_like): reservoir adjacency weight matrix, set in ``.build()``
        Win (array_like): reservoir input weight matrix, set in ``.build()``
        Wout (array_like): trained output weight matrix, set in ``.train()``
    """

    def __init__(self,
                 system_dim,
                 reservoir_dim,
                 input_dim,
                 time_dim=None,
                 num_layers=1,
                 num_groups=1,
                 local_size=None,
                 local_halo=0,
                 batch_size=None,
                 sparsity=0.95,
                 sparse_adj_matrix=False,
                 random_seed=1,
                 sigma_bias=0,
                 sigma=1,
                 leak_rate=1.0,
                 spectral_radius=1,
                 readout_method='quadratic',
                 training_method='pinv',
                 ybar=0,
                 sbar=0,
                 tikhonov_parameter=0,
                 **kwargs):

        self.system_dim = system_dim
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.local_size = local_size
        self.local_halo = local_halo
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.sparse_adj_matrix = sparse_adj_matrix
        self.spectral_radius = spectral_radius
        self.sigma_bias = sigma_bias
        self.sigma = sigma
        self.leak_rate = leak_rate
        self.readout_method = readout_method
        self.ybar = ybar
        self.sbar = sbar
        self.tikhonov_parameter = tikhonov_parameter
        self.training_method = training_method
        self._random_num_generator = np.random.default_rng(random_seed)

    def weights_init(self):
        """Initialize the weight matrices

        Notes:
            Generate the random adjacency (A) and input weight matrices (Win)
            with sparsity determined by ``sparsity`` attribute,
            scaled by ``spectral_radius`` and ``sigma`` parameters, 
            respectively. If density of A is <=0.2, then use scipy sparse
            matrices for computational speed up

        Sets Attributes:
            A (array_like): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix
            Win (array_like): (reservoir_dim, input_dim),
                reservoir input weight matrix
            Adense (array_like): stores dense version of A if A is specified
                as scipy sparse format
        """

        # Create adjacency weight matrix that defines reservoir dynamics
        # Dense version
        if not self.sparse_adj_matrix:
            # initialize weights with a random matrix centered around zero:
            A = (self._random_num_generator.random(
                    (self.reservoir_dim, self.reservoir_dim))
                 - 0.5)

            # delete the fraction of connections given by (self.sparsity):
            A[self._random_num_generator.random(A.shape) < self.sparsity] = 0

            # compute the spectral radius of self.Win, u) these weights:
#             try:
            radius = np.max(np.abs(np.linalg.eigvals(A)))
#             except:
#                 # This approach for getting the leading eigenvalue is supported
#                 # by cupy:
#                 # ISSUE: this is incorrect
#                 #       replace with appropriate method for cupy implementation
#                 _, R = np.linalg.qr(A)
#                 radius = np.abs(R[0, 0])
#                 raise Exception('ERROR: unsupported method for finding leading'
#                                 ' eigenvalue.')

            # rescale the adjacencey matrix to reach the requested spectral
            # radius:
            A = A * (self.spectral_radius / radius)

        else:
            # ---------------------- #
            # --- Sparse version --- #
            # ---------------------- #

            # stats.uniform(loc,scale) specifies uniform between [loc,
            # loc+scale]
            uniform = stats.uniform(-1.0, 2.)
            uniform.random_state = self._random_num_generator
            A = sparse.random(self.reservoir_dim, self.reservoir_dim,
                              density=(1-self.sparsity), format='csr',
                              data_rvs=uniform.rvs,
                              random_state=self._random_num_generator)

            try:
                eig = sparse.linalg.eigs(A, k=1, return_eigenvectors=False,
                                         which='LM', tol=1e-10)
            except sparse.linalg.ArpackNoConvergence as err:
                k = len(err.eigenvalues)
                if k <= 0:
                    raise AssertionError("Spurious no-eigenvalues-found case")
                eig = err.eigenvalues

            radius = np.max(np.abs(eig))
            A.data = A.data * (self.spectral_radius / radius)

        if self.sigma_bias == 0:
            # random input weights:
            Win = (self._random_num_generator.random(
                (self.reservoir_dim, self.input_dim)) * 2 - 1)
            Win = self.sigma * Win
        else:
            Win = (self._random_num_generator.random(
                (self.reservoir_dim, self.input_dim)) * 2 - 1)
            Win_input = np.ones((self.reservoir_dim, 1))
            Win = self.sigma * Win
            Win_input *= self.sigma_bias
            Win = np.hstack([Win_input, Win])

        self.A = A
        self.Win = Win
        self.Adense = A.asformat('array') if self.sparse_adj_matrix else A


    def generate(self, u, A=None, Win=None, r0=None):
        """generate reservoir time series from input signal u
        Args:
            u (array_like): (time_dimension, system_dimension), input signal to reservoir
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                                      reservoir adjacency matrix
            Win (array_like, optional): (reservoir_dim, system_dimension),
                                        reservoir input weight matrix
            r0 (array_like, optional): (reservoir_dim,) initial reservoir state
        Returns:
            r (array_like): (reservoir_dim, time_dimension), reservoir state
        """

        r = np.zeros((u.shape[0], self.reservoir_dim))

        if r0 is not None:
            logging.debug('generate:: using initial reservoir state: {}'.format(r0))
            r[0, :] = np.reshape(r0, (1, self.reservoir_dim))

        # encoding input signal {u(t)} -> {s(t)}
        for t in range(1, u.shape[0]):
            r[t, :] = self.update(r[t - 1], u[t - 1, :], A, Win)

        return r


    def update(self, r, u, A=None, Win=None):
        """update reservoir state with input signal and reservoir state at previous time step
        Args:
            r (array_like): (reservoir_dim,) reservoir state
            u (array_like): (input_dimension,) input signal
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                reservoir adjacency matrix
            Win (array_like, optional): (reservoir_dim, input_dimension),
                reservoir input weight matrix
        Returns:
            q (array_like): (reservoir_dim,) reservoir state at the next time step
        """

        if A is None:
            A = self.A
        if Win is None:
            Win = self.Win

        try:
            if self.sigma_bias != 0:
                u = np.concatenate(([1.0], u))
            p = A @ r.T + Win @ u # transposed r
            q = self.leak_rate * np.tanh(p) + (1 - self.leak_rate) * r

        except:
            print('A.shape = {}, s.shape = {}, Win.shape = {}, u.shape = {}'.format(A.shape, r.shape, Win.shape, u.shape))
            raise Exception('Likely dimension mismatch.')

        return q
        

    def predict(self, data_obj, initial_index=0, n_steps=100, spinup_steps=0,
                r0=None, keep_spinup=True):
        """Compute the prediction phase of the RC
        Args:
            dataobj (Data): data object containing the initial conditions
            initial_index (int, optional): time index of initial conditions in the data object 'values'
            n_steps (int, optional): number of steps to conduct the prediction
            spinup_steps (int, optional): number of steps before the initial_index to use for spinning up the reservoir state
            r0 (array_like, optional): initial reservoir state
        Returns:
            dataobj_pred (Data): Data object covering prediction period
        Todo:
            change steps to n_steps
        """

        # Checks
        if (initial_index > data_obj.time_dim):
            raise Exception('initial index = {}, but data_obj.time_dim = {}'.format(
                initial_index,  data_obj.time_dim))

        times = data_obj.times
        timescale = times[1]-times[0]

        # Initialize the output data object
        data_obj_pred = deepcopy(data_obj)
        data_obj_pred
        data_obj_pred.values[initial_index:, :] = 0.0

        # Recompute the initial reservoir spinup to get reservoir states
        if (spinup_steps > 0):
            u = data_obj.values[np.arange(initial_index-spinup_steps, initial_index)]
            r = self.generate(u,  r0=r0)
            r0 = r[-1, ]

        if r0 is not None:
            r_last = r0
#         else:
#             r_last = data_obj.values[initial_index-1, :]

        u_last = data_obj_pred.values[max(initial_index-1, 0), :]

        # Use these if possible
        A = getattr(self, 'A', None)
        Win = getattr(self, 'Win', None)
        predicted_values = self._predict_backend(n_steps, r_last.T, u_last.T, timescale,
                                                A=A, Win=Win)

#         data_obj_pred.values[initial_index:initial_index+n_steps, :] = predicted_values.values
        data_obj_pred.values = predicted_values.values

        if not keep_spinup:
            data_obj_pred.values = data_obj_pred.values[initial_index:]

        return data_obj_pred


    def readout(self, rt, Wout=None, utm1=None):
        """use Wout to map reservoir state to output
        Args:
            rt (array_like): 1D or 2D with dims: (Nr,) or (Ntime, Nr)
                reservoir state, either passed as single time snapshot,
                or as matrix, with reservoir dimension as last index
            utm1 (array_like): 1D or 2D with dims: (Nu,) or (Ntime, Nu)
                u(t-1) for r(t), only used if readout_method = 'biased',
                then Wout*[1, u(t-1), r(t)]=u(t)
        Returns:
            vt (array_like): 1D or 2D with dims: (Nout,) or (Ntime, Nout)
                depending on shape of input array
        Todo:
            generalize similar to DiffRC
        """

        if (Wout is None):
            Wout = self.Wout

        # necessary to copy in order to not
        # assign input reservoir state in place
        if self.readout_method=='quadratic':
            st = deepcopy(rt)
            st[...,1::2] = st[...,1::2]**2
        elif self.readout_method == 'biased':
            assert(utm1 is not None)
            if rt.ndim > 1:
                st = np.concatenate((np.ones((rt.shape[0], 1)), utm1, rt), axis=1)
            else:
                st = np.concatenate(([1.0], utm1, rt))
        else:
            st = rt

        try:
            vt = st @ Wout #.T
        except:
            print('st.shape = {}, Wout.shape = {}'.format(st.shape,Wout.shape))
            raise

        return vt

    def _predict_backend(self, n_samples, s_last, u_last, timescale,
                A=None, Win=None, Wout=None):
        """
        Apply the learned weights to new input.
        Args:
            n_samples (int): number of time steps to predict
            s_last (array_like): 1D vector with final reservoir state before prediction
            u_last (array_like): 1D vector with final input signal before prediction
            timescale (float): full time length of spinup and prediction windows
            A (array_like, optional): (reservoir_dim, reservoir_dim),
                adjacency matrix
            Win (array_like, optional): (reservoir_dim, input_dimension),
                input weight matrix
        Returns:
            y (Data): data object with predicted signal from reservoir
        """

        s = np.zeros((n_samples, self.reservoir_dim))
        y = np.zeros((n_samples, self.system_dim))

        s[0, :] = self.update(s_last, u_last, A, Win)
        y[0, :] = self.readout( s[0, :], Wout, utm1=u_last )

        for t in range(n_samples - 1):
            s[t + 1, :] = self.update(s[t, :], y[t, :], A, Win)
            y[t + 1, :] = self.readout( s[t + 1, :], Wout , utm1=y[t, :])

        y_obj = data.Data(system_dim=y.shape[1],
                     time_dim=y.shape[0],
                     values=y,
                     times=np.arange(1, y.shape[0]+1)*timescale)

        return y_obj


    def train(self, data_obj, update_Wout=True):
        """
        Train the localized RC model
        Args:
            dataobj (Data): Data object containing training data
            update_Wout (bool): if True, update Wout, otherwise
                initialize it by rewriting the toolkit's ybar and sbar matrices
        Sets Attributes:
            Wout (array_like): Trained output weight matrix
        """

        r = self.state[:, :]
        u = data_obj.values[:,  :]
        self.Wout = self._compute_Wout(r, u, update_Wout=update_Wout, u=u.T)


    def _compute_Wout(self, rt, y, update_Wout=True, u=None):
        """solve linear system with multiple RHS for readout weight matrix
        Args:
            rt (array_like): 2D with dims (time_dim, reservoir_dim),
                reservoir state
            y (array_like): 2D with dims (time_dim, output_dim),
                target reservoir output
            update_Wout (bool): if True, update Wout, otherwise,
                initialize it by rewriting the ybar and sbar matrices
        Returns:
            Wout (array_like): 2D with dims (output_dim, reservoir_dim),
                this is also stored within the object
        Sets Attributes:
            ybar (array_like): y.T @ st, st is rt with readout_method accounted for
            sbar (array_like): st.T @ st, st is rt with readout_method accounted for
            Wout (array_like): see Returns
            y_last, s_last, u_last (array_like): the last element of output, reservoir, and input states
        """
        # Prepare for nonlinear readout function:
        # necessary to copy in order to not
        # assign input reservoir state in place
        if (self.readout_method == 'quadratic'):
            st = deepcopy(rt)
            st[...,  1::2] = st[..., 1::2]**2
        elif self.readout_method == 'biased':
            assert(u is not None)
            st = np.concatenate((np.ones((rt.shape[0]-1, 1)), u[:-1, :], rt[1:, :]), axis=1)
            y = y[1:]
        else:
            st = rt

        # learn the weights by solving for final layer weights Wout analytically
        # (this is actually a linear regression, a no-hidden layer neural network with the identity activation function)
        self.ybar = self.ybar + np.dot(y.T, st)  if update_Wout else np.dot(y, st)
        self.sbar = self.sbar + np.dot(st.T, st) if update_Wout else np.dot(st.T,st)

        self.Wout = self._linsolve(self.sbar+self.tikhonov_parameter*np.eye(self.sbar.shape[0]),
                             self.ybar,
                             method=self.training_method)

        # These are from the old update_Wout method,
        # although I'm not sure what they're for
        self.y_last = y[-1,...]
        self.s_last = rt[-1,...]
        if u is not None:
            self.u_last = u[-1,...]

        return self.Wout

    def _linsolve(self, X, Y, beta = None, **kwargs):
        '''Linear solver wrapper
        Solve for A in Y = AX
        Args:
            X (matrix) : independent variable
            Y (matrix) : dependent variable
            beta (float): Tikhonov regularization
        '''

        A = self._linsolve_pinv(X, Y, beta)
        return A.T

    def _linsolve_pinv(self, X, Y, beta = None):
        """Solve for A in Y = AX, assuming X and Y are known.
        Args:
          X : independent variable, square matrix
          Y : dependent variable, square matrix
        Returns:
          A : Solution matrix, rectangular matrix
        """
        if beta is not None:
            Xinv = linalg.pinv(X+beta*np.eye(X.shape[0]))
        else:
            Xinv = linalg.pinv(X)
        print(Y.shape, Xinv.shape)
        A = Y @ Xinv

        return A
