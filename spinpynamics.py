from itertools import product
import numpy as np
from scipy.linalg import expm


class ProductOperator():
    """Cartesian product operator

    Parameters
    ----------
    comps : str, len (n_spins, )
        Labels of Cartesian components of each spin in operator - including
        'i' for identity if required.
    coef : float
        Coeficient of this operator.
    spins : array-like, shape (n_spins, )
        Total spin quantum number of each spin centre. If None (default)
        then all spins are assumed to be S = 1/2.
    
    Methods
    -------
    """

    def __init__(self, comps, *, coef=1.0, spins=None):
        self.comps = comps
        self.coef = coef
        self.spins = spins
        self._gen_matrix_rep()

    def set_coef(self, coef):
        self.coef = coef

    def get_coef(self):
        return self.coef

    def get_label(self):
        return self.as_string()

    def as_matrix(self):
        """Return matrix representation of operator scaled by coef

        Returns
        -------
        mat_rep : ndarray
        """
        return self.coef*self.mat_rep_

    def as_string(self, op_pattern=None, coef_pattern=None):
        """Generate string representation of cartesian component operator

        Parameters
        ----------
        op_pattern : str
            For formatting label, must contain a field for each component.
            If None, label is S{:}, where {:} is replaced by self.comps.
        coef_pattern : str
            For formatting coefficient. If None then coeficient is not
            included.

        Returns
        -------
        label : str
        """
        # Make operator string
        if op_pattern is None:
            op_label = 'S'.join(comp for comp in self.comps)
        else:
            op_label = op_pattern.format([comp for comp in self.comps])
        # Add amplitude if requested
        if coef_pattern is not None:
            coef_label = ''
        else:
            coef_label = coef_pattern.format(self.coef)
        # Combine and return
        return coef_label + op_label

    def _gen_matrix_rep(self):
        """Generate matrix representation"""
        # Count spins
        self.n_spins_ = len(self.comps)
        # Start matrix
        self.mat_rep_ = np.array([[1.0]])
        # Inflate with operators
        for i_spin in range(self.n_spins_):
            op_mat = self._onespin_matrix_rep(i_spin)
            self.mat_rep_ = np.kron(self.mat_rep_, op_mat)
        # Count final dimension
        self.n_hilb_ = self.mat_rep_.shape[0]
        return self

    def _onespin_matrix_rep(self, ind):
        """Returns matrix representation of specified operator

        Parameters
        ----------
        ind : int
            Index of operator to return

        Returns
        -------
        mat : ndarray
            Matrix representation of this spin operator
        """
        # Get spin
        if self.spins is None:
            spin = 1.0/2.0
        else:
            spin = self.spins[ind]
        # Get two S and Hilbert dimension
        twoS = int(2*spin)
        n_hilb = twoS + 1
        # Make array
        mat = np.zeros((n_hilb, n_hilb), dtype=np.complex128)
        azimuths = np.linspace(spin, -spin, n_hilb)
        # Make operator
        if self.comps[ind] == 'x':
            for i_ms, ms in enumerate(azimuths[:-1]):
                k = np.sqrt(spin*(spin + 1) - ms*(ms - 1))/2.0
                mat[i_ms, i_ms + 1] = k
                mat[i_ms + 1, i_ms] = k
        elif self.comps[ind] == 'y':
            for i_ms, ms in enumerate(azimuths[:-1]):
                k = np.sqrt(spin*(spin + 1) - ms*(ms - 1))/2.0j
                mat[i_ms, i_ms + 1] = k
                mat[i_ms + 1, i_ms] = -k
        elif self.comps[ind] == 'z':
            for i_ms, ms in enumerate(azimuths):
                mat[i_ms, i_ms] = ms
        elif self.comps[ind] in ['i', 'e']:
            for i_ms, ms in enumerate(azimuths):
                mat[i_ms, i_ms] = 1.0
        return mat

    def __str__(self):
        return self.as_string()


class MultiOperatorMixin():
    """Methods relating to objects containing multiple ProductOperators.
    
    Parameters
    ----------

    Methods
    -------
    """

    def get_label(self):
        return self.as_string()

    def as_matrix(self):
        """Return matrix representation of operator scaled by coef

        Returns
        -------
        mat_rep : ndarray
        """
        mat = 0.0
        # Loop through all cartesian components
        for comps in self.comps_:
            # Get operator
            op = self.ops_[comps]
            # If there is non-zero amounts of it add it to the total
            if ~np.isclose(op.coef, 0.0):
                mat += op.as_matrix()
        return mat

    def as_string(self, op_pattern=None, coef_pattern=None):
        """Generate string representation of spin operator

        Parameters
        ----------
        op_pattern : str
            For formatting label, must contain a field for each component.
            If None, label is S{:}, where {:} is replaced by self.comps.
        coef_pattern : str
            For formatting coefficient. If None then coeficient is included
            to the first decimal place.

        Returns
        -------
        label : str
        """
        label = ''
        # Loop through all cartesian components
        for comps in self.comps_:
            # Get operator
            op = self.ops_[comps]
            # If there is non-zero amounts of it add it to the total
            if ~np.isclose(op.coef, 0.0):
                label += op.as_string(op_pattern, coef_pattern='{:+.1f}')
        return label

    def _gen_all_comp_ops(self):
        """Generates all possible Cartesian component operators.
        """
        # All possible components
        comp_iter = product(['i', 'x', 'y', 'z'], repeat=self.n_spins)
        self.all_comps_ = [''.join(comps) for comps in comp_iter]
        # Make dictionary of operators with comps as keys
        self.all_ops_ = {(comps, ProductOperator(comps, spins=self.spins))
                         for comps in self.all_comps_}
        return self

    def __str__(self):
        return self.as_string()


class CartesianBasis(MultiOperatorMixin):
    """Cartesian basis set for a particular number of spins.

    As initialised, the coeffiecients of all basis operators is one. This
    should probably be maintained, if you want to change the coefficients
    use the SpinOperator class instead.

    Parameters
    ----------
    n_spins : int
        Number of spins
    spins : array-like, shape (n_spins, )
        Total spin quantum number of each spin centre. If None (default)
        then all spins are assumed to be S = 1/2.
    
    Methods
    -------

    """

    def __init__(self, n_spins, *, spins=None):
        self.n_spins = n_spins
        self.spins = spins
        self._gen_all_comp_ops()


class SpinOperator(MultiOperatorMixin):
    """Spin operator (which can comprise linear combinations of product
    operators)

    Initialised from ProductOperator objects.

    Parameters
    ----------
    ops : list of ProductOperator objects
        Operators which make up SpinOperator.
    
    Methods
    -------

    """

    def __init__(self, ops):
        self.ops = ops

    def set_coef(self, comps, coef):
        """Set coefficient of an operator

        Parameters
        ----------
        comps : str, len (n_spins, )
            Labels of Cartesian components of each spin in operator - including
            'i' for identity if required.
        coef : float
            Coefficient of operator
        """
        if comps in self.comps:
            self.ops_[comps].coef = coef
        else:
            mssg = '{:} is not a valid component string'.format(comps)
            raise ValueError(mssg)
        return self

    def get_coef(self, comps):
        """Set coefficient of an operator

        Parameters
        ----------
        comps : str, len (n_spins, )
            Labels of Cartesian components of each spin in operator - including
            'i' for identity if required.
        """
        if comps in self.comps:
            return self.ops_[comps].coef
        else:
            mssg = '{:} is not a valid component string'.format(comps)
            raise ValueError(mssg)

    def __add__(self, spin_operator):
        # Make sure same size
        pass
# class CartesianSpinOperator():
#     """Spin operator expressed as sum of Cartesian components

#     Parameters
#     ----------
#     """


# class SpinOperator():
#     """General spin operator

#     Parameters
#     ----------
#     spins : array_like
#         Spin quantum numbers
#     mat : array_like
#         Matrix representation of spin operator. If None, operator is assumed
#         to be identity on all spins.
#     labels : array-like
#         Labels of spins for printing

#     """
#     def __init__(self, spins, *, mat=None, labels=None):
#         self.spins = spins
#         self.mat = mat
#         self.labels = labels

#     def __str__(self):
#         return self.as_string(print_amp=True, print_brackets=False)

#     def __mul__(self, val):
#         if isinstance(val, SpinOperator):
#             return SpinOperator(spins=self.spins, mat=self.mat*val.mat,
#                                 labels=self.labels)
#         else:
#             self.mat = val*self.mat
#             return self
#     __rmul__ = __mul__

#     def __add__(self, val):
#         return SpinOperator(spins=self.spins, mat=self.mat + val.mat,
#                             labels=self.labels)

#     def __sub__(self, val):
#         return SpinOperator(spins=self.spins, mat=self.mat - val.mat,
#                             labels=self.labels)

#     def set_matrix_rep(self, mat):
#         """Set matrix representation

#         Parameters
#         ----------
#         mat : array-like
#             Spin operator in Hilbert space
#         """
#         self.mat = mat
#         return self

#     def get_matrix_rep(self):
#         """Get matrix representation

#         Returns
#         ----------
#         mat : ndarray
#             Spin operator in Hilbert space
#         """
#         # If specified
#         if self.mat is not None:
#             return self.mat
#         # Otherwise make identity
#         else:
#             hilb_dim = self._calc_hilb_dim()
#             return np.identity(hilb_dim)

#     def gen_from_comps(comps, *, amp=1.0, spins=None, labels=None):
#         """Make SpinOperator from cartesian components and amplitudes

#         Parameters
#         ----------
#         comps : array-like, shape (n_spins, )
#             Labels for operators (in order). These can technically be anything
#             but each element would generally be one of {'i', 'x', 'y', 'z'}.
#         amp : float
#             Amplitude of operator.
#         spins : array-like, shape (n_spins, )
#             Spin quantum number S of each spin. If None all spins are assumed
#             to be S = 1
#         labels : array-like, shape (n_spins, )
#             Labels of each spin for printing. If None a single S is inserted
#             before the components.
#         """
#         # Make spins if not given
#         if spins is None:
#             spins = [1/2 for comp in comps]
#         # Make Cartesian operator
#         op = CartesianOperator(comps, spins=spins, labels=labels)
#         # Make Spin operator
#         return SpinOperator(spins, mat=amp*op.as_matrix(), labels=labels)

#     def decompose(self):
#         """Convert spin operator into components in Cartesian basis.

#         Returns
#         -------
#         spin_ops : dict of CartesianOperators
#             All spin operators with non-zero exptation values using
#             components as keys.
#         """
#         # Empty dictionary
#         spin_ops = dict()
#         # Make cartesian basis if it doesn't already exist
#         if ~hasattr(self, 'cartesian_basis_ops'):
#             self.cartesian_basis_ops = self._gen_cartesian_basis()
#         # Loop through all operators
#         for op_label, op in self.cartesian_basis_ops.items():
#             # Get expectation value
#             amp = np.real(self.expectation_value(op))
#             # Store if amplitude not zero
#             if not np.isclose(amp, 0.0):
#                 op_sig = CartesianOperator(op.comps, amp=amp, spins=self.spins,
#                                            labels=self.labels)
#                 spin_ops[op_label] = op_sig
#         return spin_ops

#     def expectation_value(self, spin_op):
#         """Returns expecatation value of spin_op

#         Parameters
#         ----------
#         spin_op : SpinOperator
#             Operator to take expectation value with respect to.

#         Returns
#         -------
#         val : float
#             Expecatation value
#         """
#         # Matrix representations
#         self_mat = self.mat
#         spin_mat = spin_op.as_matrix()
#         # Get normalisation constant
#         norm = np.trace(np.matmul(spin_mat, spin_mat))
#         # Get expectation value
#         val = np.trace(np.matmul(self_mat, spin_mat))/norm
#         return val

#     def nutate(self, nut_op, angle):
#         """Rotation of spin operator about nut_op.

#         Returns a copy of the SpinOperator object rather than modifying the
#         existing object.

#         Parameters
#         ----------
#         nut_op : SpinOperator
#             SpinOperator about which rotation is applied
#         angle : float
#             Angle of rotation in degrees
#         """
#         # Copy obect
#         rho = SpinOperator(spins=self.spins, mat=self.as_matrix(),
#                            labels=self.labels)
#         # If angle is zero return copy without rotation
#         if np.isclose(angle, 0.0):
#             return rho
#         # Calculate commutator
#         com = self.commutator(nut_op)
#         # If commutator is zero there is no rotation
#         if np.all(np.isclose(com.mat, 0.0)):
#             return rho

#         # Otherwise if commutator is orthogonal can still avoid matrix expon.
#         com_ops = com.decompose()
#         if len(com_ops) == 1:
#             # Get commutator operator
#             com_op = com_ops[list(com_ops.keys())[0]]
#             # Reset matrix representation
#             rho.mat = 0.0
#             # Calculate circular decomposition
#             cos_ang = np.cos(angle*np.pi/180.0)
#             sin_ang = np.sin(angle*np.pi/180.0)
#             # Generate rotated matrix representation
#             rho.mat += cos_ang*self.as_matrix()
#             rho.mat += sin_ang*com_op.as_matrix()
#         # Last resort is to brute force using propogator
#         else:
#             prop_arg = -(1.0j*angle*np.pi/180.0)*nut_op.as_matrix()
#             prop = expm(prop_arg)
#             prop_t = expm(-1.0*prop_arg)
#             rho.mat = np.matmul(np.matmul(prop, self.mat), prop_t)
#         return rho

#     def commutator(self, spin_op):
#         """Returns commutator of self with spin_op as a separate SpinOperator

#         Parameters
#         ----------
#         spin_op: SpinOperator

#         Returns
#         -------
#         com_op: SpinOperator
#         """
#         # Calculatuate commutator with nutation operator
#         opA = self.as_matrix()
#         opB = spin_op.as_matrix()
#         com_mat = np.matmul(opA, opB) - np.matmul(opB, opA)
#         return SpinOperator(self.spins, mat=com_mat, labels=self.labels)

#     def as_matrix(self):
#         """See get_matrix_rep"""
#         return self.get_matrix_rep()

#     def as_string(self, print_amp=False, print_brackets=False):
#         """String representation of decomposed density matrix

#         Parameters
#         ----------
#         print_amp : bool
#             If True, include amplitude in string e.g. '+1.0SxSy'
#         print_brackets : bool
#             If True, include brackets in string e.g. '<SxSy>'

#         Returns
#         -------
#         rho_str : str
#             Human readable string representation of operator
#         """
#         # Decompose into operators
#         spin_ops = self.decompose()
#         # Blank string
#         rho_str = ''
#         # Add string of each operator to total
#         for op_label, op in spin_ops.items():
#             if print_amp is True:
#                 op_label = '{:+.1f}'.format(op.amp) + op_label
#             rho_str += op_label

#         return rho_str

#     def _calc_hilb_dim(self):
#         """Calculate Hilbert dimension"""
#         hilbert_dim = 1
#         for spin in self.spins:
#             hilbert_dim *= (int(2*spin) + 1)
#         return hilbert_dim

#     def _gen_cartesian_basis(self):
#         """Generate complete cartesian basis for these spins with unit
#         amplitudes

#         Returns
#         -------
#         Observerables object
#         """
#         # Make list of all cartesian components
#         all_comps = (product(['i', 'x', 'y', 'z'], repeat=len(self.spins)))
#         all_comps = [''.join(comp) for comp in all_comps]
#         # Make list of operators
#         ops = [CartesianOperator(comp, spins=self.spins, labels=self.labels)
#                for comp in all_comps]
#         # Return as Observable
#         return Observables(ops)


# class CartesianOperator():
#     """Cartesian spin operator

#     Differs from SpinOperator as can only be one Cartesian spin operator
#     with unit amplitude. Mostly used for making SpinOperators and extracting
#     expectation values.

#     Parameters
#     ----------
#     comps : array-like, shape (n_spins, )
#         Labels for operators (in order). These can technically be anything
#         but each element would generally be one of {'i', 'x', 'y', 'z'}.
#     amp : float
#         Amplitude of the operator, default 1.0.
#     spins : array-like, shape (n_spins, )
#         Spin quantum number S of each spin. If None all spins are assumed to
#         be S = 1/2
#     labels : array-like, shape (n_spins, )
#         Labels of each spin for printing. If None a single S is inserted before
#         the components.
#     """
#     def __init__(self, comps, *, amp=1.0, spins=None, labels=None):
#         self.comps = comps
#         self.amp = amp
#         self.spins = spins
#         self.labels = labels

#     def __str__(self):
#         return self.as_string(print_brackets=False)

#     def get_matrix_rep(self):
#         """Get matrix representation

#         Returns
#         ----------
#         mat : ndarray
#             Spin operator in Hilbert space
#         """
#         # Start with 1.0
#         mat = np.array([[1.0]])
#         # Inflate with operators
#         for i_comp, _ in enumerate(self.comps):
#             mat = np.kron(mat, self._onespin_matrix_rep(i_comp))
#         # Return with amplitude included
#         return self.amp*mat

#     def as_string(self, print_amp=False, print_brackets=False):
#         """String representation of operator

#         Parameters
#         ----------
#         print_amp : bool
#             If True, include amplitude in string e.g. '+1.0SxSy'
#         print_brackets : bool
#             If True, include brackets in string e.g. '<SxSy>'

#         Returns
#         -------
#         op_label : str
#             Human readable string representation of operator
#         """
#         # Make plain operator
#         # Include labels if given
#         if self.labels is not None:
#             op_label = ''
#             for i_comp, comp in enumerate(self.comps):
#                 op_label += '{:}{:}'.format(self.labels[i_comp], comp)
#         # Otherwise leave plain
#         else:
#             op_label = 'S'
#             for comp in self.comps:
#                 op_label += '{:}'.format(comp)
#         # Add brackets if requested
#         if print_brackets is True:
#             op_label = r'$\langle$' + op_label + r'$\rangle$'
#         # Add amplitude if requested
#         if print_amp is True:
#             op_label = '{:+.1f}'.format(self.amp) + op_label
#         return op_label

#     def as_matrix(self):
#         return self.get_matrix_rep()

#     def _onespin_matrix_rep(self, ind):
#         """Returns matrix representation of specified operator

#         Parameters
#         ----------
#         ind : int
#             Index of operator to return

#         Returns
#         -------
#         mat : ndarray, shape (n_hilb, n_hilb)
#         """
#         # Get two*S
#         if self.spins is None:
#             twoS = 1
#         else:
#             twoS = int(2*self.spins[ind])
#         # Other relevant constants
#         n_hilb = twoS + 1
#         spin = twoS / 2.0
#         # Make array
#         mat = np.zeros((n_hilb, n_hilb), dtype=np.complex128)
#         azimuths = np.linspace(twoS/2.0, -twoS/2.0, n_hilb)
#         # Make operator
#         if self.comps[ind] == 'x':
#             for i_ms, ms in enumerate(azimuths[:-1]):
#                 k = np.sqrt(spin*(spin + 1) - ms*(ms - 1))/2.0
#                 mat[i_ms, i_ms + 1] = k
#                 mat[i_ms + 1, i_ms] = k
#         elif self.comps[ind] == 'y':
#             for i_ms, ms in enumerate(azimuths[:-1]):
#                 k = np.sqrt(spin*(spin + 1) - ms*(ms - 1))/2.0j
#                 mat[i_ms, i_ms + 1] = k
#                 mat[i_ms + 1, i_ms] = -k
#         elif self.comps[ind] == 'z':
#             for i_ms, ms in enumerate(azimuths):
#                 mat[i_ms, i_ms] = ms
#         elif self.comps[ind] in ['i', 'e']:
#             for i_ms, ms in enumerate(azimuths):
#                 mat[i_ms, i_ms] = 1.0
#         return mat


# class Observables(dict):
#     """Dictionary with spin operator labels as keys and operators as values

#     Parameters
#     ----------
#     operators : array-like of SpinOperator
#         Spin operators to store
#     """

#     def __init__(self, operators):
#         key_list = [op.as_string() for op in operators]
#         val_list = [op for op in operators]
#         super().__init__(zip(key_list, val_list))

#     def get_amps(self):
#         amp_dict = {}
#         for key, val in self.items():
#             amp_dict[key] = val.amp
#         return amp_dict

#     def set_amps(self, amps):
#         for key, amp in amps.items():
#             self[key] = amp

#     def gen_from_comps(comps, amps, spins, labels):
#         """Generate observables from list of components

#         Parameters
#         ----------
#         comps : array-like of str, shape (n_ops, )
#             List of components
#         amps : array-like of float, shape (n_ops, )
#             List of amplitudes
#         spins : array-like, shape (n_spins, )
#             Spin quantum number S of each spin. If None the number of spins
#             is determined from comps and are all assumed to be S = 1/2.
#         labels : array-like of str, shape (n_spins, )
#             Labels of each spin for printing. If None a single S is inserted
#             before the components.

#         Returns
#         -------
#         Observerables object
#         """
#         # Make list of operators
#         ops = [SpinOperator.gen_from_comps(comp, amp=amp, spins=spins,
#                                            labels=labels)
#                for (comp, amp) in zip(comps, amps)]
#         # Return as Observable
#         return Observables(ops)




# if __name__ == "__main__":
#     op = SpinOperator.gen_from_comps('z')
#     nut_op = (np.cos(np.pi/4)*SpinOperator.gen_from_comps('y') -
#               np.sin(np.pi/4)*SpinOperator.gen_from_comps('x'))
#     op = op.nutate(nut_op, 90.0)
#     print(op)