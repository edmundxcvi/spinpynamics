from itertools import product
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


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
    """

    def __init__(self, comps, *, coef=1.0, spins=None):
        self.comps = comps
        self.coef = coef
        self.spins = spins
        self._gen_matrix_rep()

    def set_coef(self, coef):
        self.coef = coef
        return self

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
            op_label = 'S' + ''.join([comp for comp in self.comps])
        else:
            op_label = op_pattern.format([comp for comp in self.comps])
        # Add amplitude if requested
        if coef_pattern is None:
            coef_label = ''
        else:
            coef_label = coef_pattern.format(self.coef)
        # Combine and return
        return coef_label + op_label

    def commutator(self, op, *, norm=False):
        """Calculate commutator with another product operator

        op is treated as 'operator B'

        Parameters
        ----------
        op : ProductOperator
        norm : bool
            If True, commutator returned with coef=1.0
        Returns
        -------
        op_t : ProductOperator
            Commutator
        """
        # Calculate commutator
        opA = self.as_matrix()
        opB = op.as_matrix()
        com_mat = np.matmul(opA, opB) - np.matmul(opB, opA)
        # If zero return None
        if np.all(np.isclose(com_mat, 0.0)):
            return None
        else:
            # Otherwise make spin operator
            spin_op = SpinOperator.from_mat(com_mat, spins=self.spins)
            if len(spin_op.ops) == 1:
                # Ensure unit amplitude
                op_t = list(spin_op.ops.values())[0]
                if norm is True:
                    op_t.coef = op_t.coef/np.abs(op_t.coef)
                return op_t
            else:
                raise ValueError('Product operator commutator invalid.')

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

    def __add__(self, op):
        if isinstance(op, ProductOperator):
            if op.comps == self.comps:
                self.set_coef(self.coef + op.coef)
                return self
            else:
                return SpinOperator([self, op])
        if isinstance(op, SpinOperator):
            return SpinOperator([self] + list(op.ops.values()))

    def __sub__(self, op):
        if isinstance(op, ProductOperator):
            if op.comps == self.comps:
                self.set_coef(self.coef - op.coef)
                return self
            else:
                op.coef *= -1.0
                return SpinOperator([self, op])
        if isinstance(op, SpinOperator):
            all_ops = [self]
            for comps in op.ops.keys():
                op = op.ops[comps]
                op.set_coef(-1.0*op.get_coef())
                all_ops.append()
            return SpinOperator(all_ops)

    def __mult__(self, val):
        self.set_coef(val*self.coef)
        return self


class SpinOperator():
    """Spin operator (which can comprise linear combinations of product
    operators)

    Initialised from ProductOperator objects.

    Parameters
    ----------
    ops : array-like of ProductOperator objects
        Operators which make up SpinOperator.

    """

    def __init__(self, ops):
        # Get number of spins
        self.n_spins_ = ops[0].n_spins_
        # Start product operator dictionary
        self.ops = {ops[0].comps: ops[0]}
        # Add in any further product operators
        for op in ops[1:]:
            self._add_prodop(op)

    def get_label(self):
        return self.as_string()

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

    def get_coef_dict(self):
        return {(comps, op.coef) for comps, op in self.ops.items()}

    def get_coef_list(self):
        return [op.coef for op in self.ops.values()]

    def get_ops_dict(self):
        return self.ops

    def get_ops_list(self):
        return list(self.ops.values())

    def as_matrix(self):
        """Return matrix representation of operator scaled by coef

        Returns
        -------
        mat_rep : ndarray
        """
        # Set default return
        mat = None
        # Loop through all cartesian components
        for op in self.ops.values():
            # If there is non-zero amounts of it add it to the total
            if ~np.isclose(op.coef, 0.0):
                # If matrix already started
                if mat is not None:
                    # Add to total
                    mat += op.as_matrix()
                # Otherwise start matrix
                else:
                    mat = 0.0*op.as_matrix() + op.as_matrix()
        return mat

    def as_string(self, op_pattern=None, coef_pattern='{:+.1f}'):
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
        # Get all Cartesian components
        all_comps = [op.comps for op in self.ops.values()]
        all_comps.sort()
        # Start label
        label = ''
        # Loop through all cartesian components
        for comps_ in all_comps:
            # Get operator
            op = self.ops[comps_]
            # If there is non-zero amounts of it add it to the total
            if ~np.isclose(op.coef, 0.0):
                label += op.as_string(op_pattern, coef_pattern=coef_pattern)
        return label

    def normalise(self):
        """Normalises amplitudes of operator coeficients.

        Parameters
        ----------
        self

        Returns
        -------
        self
        """
        # Get coeficients
        coef_array = np.array(self.get_coef_list())
        # Get normalisation constants
        euc_norm = np.sqrt(np.sum(coef_array*coef_array))
        # Normalise array
        coef_norm = coef_array*np.sqrt(2.0)/euc_norm
        # Update coeficients
        for coef_, op in zip(coef_norm, self.ops.values()):
            op.set_coef(coef_)
        return self

    def expectation_value(self, op):
        """Get expectation value of self with respect to op

        Parameters
        ----------
        op : SpinOperator or ProductOperator

        Returns
        -------
        val : float
            Expectation value
        """
        val = np.trace(np.matmul(self.as_matrix(), op.as_matrix()))
        norm = np.trace(np.matmul(op.as_matrix(), op.as_matrix()))
        return val/norm

    def nutate(self, rot_op):
        """Brute force nutation of spin operator

        Parameters
        ----------
        rot_op : ProductOperator or SpinOperator
            Operator to nutate by

        Returns
        -------
        op_t : SpinOperator
            Rotated spin operator, as a new copy of the original spin operator
        """
        # If op is a product operator
        if isinstance(rot_op, ProductOperator):
            op_t = self._nut_prodop(rot_op)
        # Otherwise if it's a spin operator
        elif isinstance(rot_op, SpinOperator):
            # If all components commute
            if rot_op._all_commute() is True:
                # Copy self
                op_t = SpinOperator(self.get_ops_list())
                # Loop through all components of rotation operator
                for op in rot_op.get_ops_list():
                    # And apply them in turn (order not important)
                    op_t = op_t._nut_prodop(op)
            # Otherwise brute force with propogators
            else:
                op_t = self._nut_propagator(rot_op)
        return op_t

    def _nut_prodop(self, rot_op):
        """Nutate by product operator using commutators

        Parameters
        ----------
        rot_op : ProductOperator
            Operator to nutate by

        Returns
        -------
        op_t : SpinOperator
            Rotated spin operator, as a new copy of the original spin operator
        """
        # Loop through all ProductOperators
        op_t = None
        for op in self.ops.values():
            # Make copy of operators with unit coeficients
            op_ = ProductOperator(op.comps, spins=op.spins)
            rot_ = ProductOperator(rot_op.comps, spins=rot_op.spins)
            # Calculate commutator
            com = op_.commutator(rot_, norm=True)
            # If commutator is not zero
            if com is not None:
                # Commutator sign
                com_sign = np.real(-com.coef/1j)
                coef_a = np.cos(rot_op.coef)*op.coef
                coef_b = np.sin(rot_op.coef)*op.coef*com_sign
                # Create operators
                op_a = ProductOperator(op.comps, coef=coef_a, spins=op.spins)
                op_b = ProductOperator(com.comps, coef=coef_b, spins=op.spins)
                # Add to temporary if significant
                if ~np.isclose(coef_a, 0.0) and ~np.isclose(coef_b, 0.0):
                    op_ = op_a + op_b
                elif ~np.isclose(coef_a, 0.0) and np.isclose(coef_b, 0.0):
                    op_ = op_a
                elif np.isclose(coef_a, 0.0) and ~np.isclose(coef_b, 0.0):
                    op_ = op_b
            # If commutator is zero
            else:
                op_ = ProductOperator(op.comps, coef=op.coef, spins=op.spins)
            # Add to total
            if op_t is None:
                op_t = op_
            else:
                op_t = op_t + op_
        # Ensure spin operator
        if isinstance(op_t, ProductOperator):
            op_t = SpinOperator([op_t])
        return op_t

    def _nut_propagator(self, rot_op):
        """Nutate by product operator using propogators

        Parameters
        ----------
        rot_op : ProductOperator or SpinOperator
            Operator to nutate by

        Returns
        -------
        op_t : SpinOperator
            Rotated spin operator, as a new copy of the original spin operator
        """
        # Calculate propogators (slow)
        prop = expm(-1.0j*rot_op.as_matrix())
        prop_t = expm(+1.0j*rot_op.as_matrix())
        # Matrix representation of rotated operator
        op_t_mat = np.matmul(np.matmul(prop, self.as_matrix()), prop_t)
        # Convert to ProductOperators
        return SpinOperator.from_mat(op_t_mat)

    def _all_commute(self):
        """Determine if all operators in SpinOperator commute

        Returns
        -------
        tf : bool
            True if all ProductOperators in SpinOperator commute
        """
        for i_opA, opA in enumerate(self.ops.values()):
            subsequent_ops = list(self.ops.values())[i_opA + 1:]
            for opB in subsequent_ops:
                if opB.commutator(opA) is not None:
                    return False
        return True

    def _add_prodop(self, op, mod=1.0):
        """Add a cartesian product operator

        Parameters
        ----------
        op : ProductOperator
            Number of spins must be consistent with self
        mod : float
            Multiplied onto coef of op, set to -1.0 for subtraction.

        Returns
        -------
        self
        """
        # If correct size
        if op.n_spins_ == self.n_spins_:
            # If operator already present
            if op.comps in self.ops:
                # Add to component
                self.ops[op.comps].coef += mod*op.coef
            # Otherwise make a new element
            else:
                self.ops[op.comps] = op
                self.ops[op.comps].set_coef(mod*op.coef)
        # If not correct size make error
        else:
            mssg = "Trying to add operators with {} and {} spins is not " + \
                   "possible. Please check."
            raise ValueError(mssg.format(op.n_spins_, self.n_spins_))
        return self

    def _add_spinop(self, op, mod=1.0):
        """Add spin operator to spin operator

        Parameters
        ----------
        op : SpinOperator
            Number of spins must be consistent with self
        mod : float
            Multiplied onto coefs of each ProductOperator of op, set to -1.0
            for subtraction.

        Returns
        -------
        self
        """
        # Copy operator
        return SpinOperator(list(self.ops.values()) + list(op.ops.values()))

    def __add__(self, op):
        if isinstance(op, ProductOperator):
            self._add_prodop(op)
            return self
        elif isinstance(op, SpinOperator):
            op_ = self._add_spinop(op)
            return op_

    def __sub__(self, op):
        if isinstance(op, ProductOperator):
            self._add_prodop(op, mod=-1.0)
        elif isinstance(op, SpinOperator):
            self._add_spinop(op, mod=-1.0)
        return self

    def __str__(self):
        return self.as_string()

    def from_mat(mat, *, spins=None):
        """Generate spin operator from matrix representation.

        Parameters
        ----------
        mat: array-like
            Matrix representation of operator
        spins : array-like, shape (n_spins, )
            Total spin quantum number of each spin centre. If None (default)
            then all spins are assumed to be S = 1/2.
        """
        # Get number of spins
        if spins is None:
            n_spins = int(np.log2(mat.shape[0]))
            # spins = np.ones((n_spins,))/2.0
        else:
            n_spins = len(spins)
        # Get components of all Cartesian product operators as strings
        all_comps = (product(['i', 'x', 'y', 'z'], repeat=n_spins))
        all_comps = [''.join(comp) for comp in all_comps]
        # Loop through all components
        ops = []
        for comps in all_comps:
            # Make product operator
            op = ProductOperator(comps, spins=spins)
            # Get matrix representation
            op_mat = op.as_matrix()
            # Get normalisation constant
            norm = np.trace(np.matmul(op_mat, op_mat))
            # Get expectation value
            coef = np.trace(np.matmul(mat, op_mat))/norm
            # Store if non-zero
            if ~np.isclose(coef, 0.0):
                op.set_coef(coef)
                ops.append(op)
        return SpinOperator(ops)


class Pulse(SpinOperator):
    """Generates SpinOpertor-like object

    Parameters
    ----------
    phase : float
        Phase of pulse in **degrees**, i.e. 0 for +x, 90 for +y, 180 for -x,
        270 for -y.
    beta : float
        Flip angle in **degrees**, i.e. 90 for pi/2 pulse and 180 for pi pulse
    active_spins : array-like of bool
        Spins which pulse acts on
    spins : array-like, shape (n_spins, )
        Total spin quantum number of each spin centre. If None (default)
        then all spins are assumed to be S = 1/2.
    """

    def __init__(self, phase, beta, active_spins, *, spins=None):
        self.phase = phase
        self.beta = beta
        self.active_spins = active_spins
        self._get_coefs()
        super().__init__(self._get_operators(spins))

    def _get_coefs(self):
        degree = 180.0/np.pi
        self.coef_x = np.cos(self.phase/degree)*self.beta/degree
        self.coef_y = np.sin(self.phase/degree)*self.beta/degree
        return self

    def _get_operators(self, spins):
        op_list = []
        for i_spin, spin in enumerate(self.active_spins):
            if spin:
                comps = ['i' for spin in self.active_spins]
                comps[i_spin] = 'p'
                comps = ''.join(comp for comp in comps)
                comps_x = comps.replace('p', 'x')
                comps_y = comps.replace('p', 'y')
                # x-pulse
                if ~np.isclose(self.coef_x, 0.0):
                    op_list.append(ProductOperator(comps_x,
                                                   coef=self.coef_x,
                                                   spins=spins))
                # y-pulse
                if ~np.isclose(self.coef_y, 0.0):
                    op_list.append(ProductOperator(comps_y,
                                                   coef=self.coef_y,
                                                   spins=spins))
        return op_list


if __name__ == '__main__':
    wA = 0.5
    wB = -1.0
    wAB = 0.1
    x_90 = Pulse(0, 90, [True, True])
    x_180 = Pulse(0, 180, [True, True])
    X = np.linspace(0, 20*np.pi, 101)
    sig = np.zeros((X.size, 4))
    rho0 = SpinOperator([ProductOperator('zi'), ProductOperator('iz')])
    rho0plus = rho0.nutate(x_90)

    for i in range(X.size):
        t = X[i]
        tau = SpinOperator([ProductOperator('zi', coef=wA*t),
                            ProductOperator('iz', coef=wB*t),
                            ProductOperator('zz', coef=wAB*t)])
        rhotauminus = rho0plus.nutate(tau)
        rhotauplus = rhotauminus.nutate(x_180)
        rhotwotau = rhotauplus.nutate(tau)
        sig[i, 0] = np.real(rhotwotau.expectation_value(ProductOperator('xz')))
        sig[i, 1] = np.real(rhotwotau.expectation_value(ProductOperator('yi')))
        sig[i, 2] = np.real(rhotwotau.expectation_value(ProductOperator('zx')))
        sig[i, 3] = np.real(rhotwotau.expectation_value(ProductOperator('iy')))
    fig, ax = plt.subplots()
    ax.plot(X, sig[:, 0], label='Sxz')
    ax.plot(X, sig[:, 1], label='Syi')
    ax.legend()
    fig, ax = plt.subplots()
    ax.plot(X, sig[:, 2], label='Szx')
    ax.plot(X, sig[:, 3], label='Siy')
    ax.legend()
    plt.show()
