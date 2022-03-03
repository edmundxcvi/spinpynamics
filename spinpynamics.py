from itertools import product
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def _get_all_comps(n_spins):
    """Makes list of all possible Cartesian components

    Parameters
    ----------
    n_spins : int
        Number of spins

    Returns
    -------
    all_comps : list of str
        All Cartesian components
    """
    all_comps = (product(['i', 'x', 'y', 'z'], repeat=n_spins))
    return [''.join(comp) for comp in all_comps]


class PropogatorMixin():

    def nutate(self, rot_op, *, force_propogator=False):
        """Rotate spinoperator about rot_op

        Parameters
        ----------
        rot_op : ProductOperator or SpinOperator
            Operator to nutate by
        force_propogator : bool
            Force use of matrix exponentials

        Returns
        -------
        op_t : SpinOperator
            Rotated spin operator, as a new copy of the original spin operator
        """
        # If using commutators
        if force_propogator is False:
            # If op is a product operator
            if isinstance(rot_op, ProductOperator):
                op_t = self._nutate_prodop(rot_op)
            # Otherwise if it's a spin operator
            elif isinstance(rot_op, SpinOperator):
                # If all components commute
                if rot_op._all_commute() is True:
                    # Rotate using commuation relationships
                    op_t = self._nutate_spinop(rot_op)
                # Otherwise brute force with propogators
                else:
                    op_t = self._nutate_propagator(rot_op)
        else:
            op_t = self._nutate_propagator(rot_op)
        return op_t

    def _nutate_propagator(self, rot_op):
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


class ProductOperator(PropogatorMixin):
    """Cartesian product operator

    Parameters
    ----------
    comps : str, len (n_spins, )
        Labels of Cartesian components of each spin in operator - including
        'i' for identity if required.
    coef : float
        coefficient of this operator.
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

    def as_matrix(self, *, norm=False):
        """Return matrix representation of operator scaled by coef

        Parameters
        ----------
        norm : bool
            If True returns normalised matrix representation, otherwise returns
            scaled by coef.
        Returns
        -------
        mat_rep : ndarray
        """
        if norm:
            return self._mat_rep
        else:
            return self.coef*self._mat_rep

    def as_string(self, op_pattern=None, coef_pattern=None):
        """Generate string representation of cartesian component operator

        Parameters
        ----------
        op_pattern : str
            For formatting label, must contain a field for each component.
            If None, label is S{:}, where {:} is replaced by self.comps.
        coef_pattern : str
            For formatting coefficient. If None then coefficient is not
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

    def commutator(self, op, *, norm=True):
        """Calculate commutator with another product operator

        op is treated as 'operator B'

        Parameters
        ----------
        op : ProductOperator
        norm : bool
            If True, commutator is returned with unit coefficient.
        Returns
        -------
        op_t : ProductOperator
            Commutator
        """
        # Calculate commutator
        opA = self.as_matrix(norm=norm)
        opB = op.as_matrix(norm=norm)
        com_mat = np.matmul(opA, opB) - np.matmul(opB, opA)
        # If zero return None
        if np.all(np.isclose(com_mat, 0.0)):
            return None
        # Otherwise make spin operator
        else:
            # Make SpinOperator
            spin_op = SpinOperator.from_mat(com_mat, spins=self.spins)
            # If comprises a single ProductOperator
            if spin_op._is_prodop():
                # Convert to ProductOperator
                op_t = spin_op.as_prodop()
                return op_t
            # Otherwise complain
            else:
                raise ValueError('Product operator commutator invalid.')

    def _nutate_prodop(self, rot_op):
        """Nutate about another ProductOperator

        Parameters
        ----------
        rot_op : ProductOperator
            Operator to nutate around

        Returns
        -------
        op_t : ProductOperator or SpinOperator
            Rotated copy of self
        """
        # Copy self
        op_t = ProductOperator(comps=self.comps, coef=self.get_coef(),
                               spins=self.spins)
        # Get commutator
        com_op = op_t.commutator(rot_op)
        # If commutator is zero return copy
        if com_op is None:
            return op_t
        # If rotation occured
        else:
            # Get coefficients in rotated operator
            coef_a = op_t.get_coef()*np.cos(rot_op.coef)
            coef_b = op_t.get_coef()*np.sin(rot_op.coef)
            # Update operators
            op_t.set_coef(coef_a)
            com_op.set_coef(-coef_b*np.real(com_op.get_coef()/1.0j))
            # If rotation is 90 + n*180 degrees
            if np.isclose(op_t.get_coef(), 0.0):
                return com_op
            # If rotation is n*180 degrees
            elif np.isclose(com_op.get_coef(), 0.0):
                return op_t
            # Otherwise
            else:
                return op_t + com_op

    def _nutate_spinop(self, rot_op):
        """Nutate about a SpinOperator

        Parameters
        ----------
        rot_op : SpinOperator
            Operator to nutate around, all components must commute.

        Returns
        -------
        op_t : ProductOperator or SpinOperator
            Rotated copy of self
        """
        # Copy self
        op_t = ProductOperator(comps=self.comps, coef=self.get_coef(),
                               spins=self.spins)
        # If all components of rot_op commute apply in turn
        if rot_op._all_commute():
            # Loop through all components
            for op_ in rot_op.get_ops_list():
                # Nutate operator
                op_t = op_t.nutate(op_)
        else:
            mssg = 'To nutate using commutators all components of the '
            mssg += 'rotation operator must commute.'
            raise ValueError(mssg)
        return op_t

    def _gen_matrix_rep(self):
        """Generate matrix representation"""
        # Count spins
        self._n_spins = len(self.comps)
        # Start matrix
        self._mat_rep = np.array([[1.0]])
        # Inflate with operators
        for i_spin in range(self._n_spins):
            op_mat = self._onespin_matrix_rep(i_spin)
            self._mat_rep = np.kron(self._mat_rep, op_mat)
        # Count final dimension
        self._n_hilb = self._mat_rep.shape[0]
        # Normalise
        self._scale = np.trace(np.matmul(self._mat_rep, self._mat_rep))
        self._scale = 1.0/np.sqrt(self._scale)
        self._mat_rep *= self._scale
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

    def _add_prodop(self, op, *, mod=1.0):
        if op.comps == self.comps:
            self.set_coef(self.get_coef() + mod*op.get_coef())
            return self
        else:
            op.set_coef(mod*op.get_coef())
            return SpinOperator([self, op])

    def _add_spinop(self, op, *, mod=1.0):
        all_ops = [self]
        for prod_op in op.get_ops_list():
            prod_op.set_coef(mod*prod_op.get_coef())
            all_ops.append(prod_op)
        return SpinOperator(all_ops)

    def __str__(self):
        return self.as_string()

    def __add__(self, op):
        if isinstance(op, ProductOperator):
            return self._add_prodop(op)
        else:
            return self._add_spinop(op)

    def __sub__(self, op):
        if isinstance(op, ProductOperator):
            return self._add_prodop(op, mod=-1.0)
        else:
            return self._add_spinop(op, mod=-1.0)

    def __mult__(self, val):
        self.set_coef(val*self.coef)
        return self


class SpinOperator(PropogatorMixin):
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
        self._n_spins = ops[0]._n_spins
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

    def get_comps_list(self):
        return list(self.ops.keys())

    def get_coef_dict(self):
        return {(comps, op.coef) for comps, op in self.ops.items()}

    def get_coef_list(self):
        return [op.coef for op in self.get_ops_list()]

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
        for op in self.get_ops_list():
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
            For formatting coefficient. If None then coefficient is included
            to the first decimal place.

        Returns
        -------
        label : str
        """
        # Get all Cartesian components
        all_comps = [op.comps for op in self.get_ops_list()]
        # all_comps.sort()
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

    def as_prodop(self):
        """Return SpinOperator as ProductOperator

        Only works if SpinOperator only contains one Product Operator.

        Returns
        -------
        op : ProductOperator
        """
        if self._is_prodop():
            return self.get_ops_list()[0]
        else:
            mssg = 'SpinOperator cannot be converted to ProductOperator ' + \
                   'as it contains {} Cartesian components.'
            raise ValueError(mssg.format(len(self.ops)))

    def normalise(self):
        """Normalises amplitudes of operator coefficients.

        Parameters
        ----------
        self

        Returns
        -------
        self
        """
        # Get coefficients
        coef_array = np.array(self.get_coef_list())
        # Get normalisation constants
        euc_norm = np.sqrt(np.sum(coef_array*coef_array))
        # Normalise array
        coef_norm = coef_array*np.sqrt(2.0)/euc_norm
        # Update coefficients
        for coef_, op in zip(coef_norm, self.get_ops_list()):
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
        return np.real(np.trace(np.matmul(self.as_matrix(), op.as_matrix())))

    def _nutate_prodop(self, rot_op):
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
        # Start output
        op_t = None
        # Loop through all ProductOperators in self
        for op in self.get_ops_list():
            # Rotate operator
            op_ = op.nutate(rot_op)
            # Add to total
            if op_t is None:
                op_t = op_
            else:
                op_t = op_t + op_
        return op_t

    def _nutate_spinop(self, rot_op):
        """Nutate by spin operator using commutators

        Parameters
        ----------
        rot_op : SpinOperator
            Operator to nutate by

        Returns
        -------
        op_t : SpinOperator
            Rotated spin operator, as a new copy of the original spin operator
        """
        # Copy self
        op_t = SpinOperator(self.get_ops_list())
        # If all components of rot_op commute apply in turn
        if rot_op._all_commute():
            # Loop through all ProductOperators in rot_op
            for op in rot_op.get_ops_list():
                # Rotate operator
                op_t = op_t._nutate_prodop(op)
        else:
            mssg = 'To nutate using commutators all components of the '
            mssg += 'rotation operator must commute.'
            raise ValueError(mssg)
        return op_t

    def _all_commute(self):
        """Determine if all operators in SpinOperator commute

        Returns
        -------
        tf : bool
            True if all ProductOperators in SpinOperator commute
        """
        for i_opA, opA in enumerate(self.get_ops_list()):
            subsequent_ops = self.get_ops_list()[i_opA + 1:]
            for opB in subsequent_ops:
                if opB.commutator(opA) is not None:
                    return False
        return True

    def _is_prodop(self):
        """Determine if SpinOperator contains more than one product operator

        Returns
        -------
        tf : bool
            True if SpinOperator contains only one ProductOperator
        """
        if len(self.ops) == 1:
            return True
        else:
            return False

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
        if op._n_spins == self._n_spins:
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
            raise ValueError(mssg.format(op._n_spins, self._n_spins))
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
        op_t = SpinOperator(list(self.get_ops_list()))
        # Add each spin operator in turn
        for op_ in op.get_ops_list():
            op_.set_coef(mod*op_.get_coef())
            op_t = op_t + op_
        return op_t

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
        else:
            n_spins = len(spins)
        # Get components of all Cartesian product operators as strings
        all_comps = _get_all_comps(n_spins)
        # Loop through all components
        ops = []
        for comps in all_comps:
            # Make product operator
            op = ProductOperator(comps, spins=spins)
            # Get matrix representation
            op_mat = op.as_matrix()
            # Get expectation value
            coef = np.trace(np.matmul(mat, op_mat))
            # Store if non-zero
            if ~np.isclose(coef, 0.0):
                op.set_coef(coef)
                ops.append(op)
        return SpinOperator(ops)


class Pulse(SpinOperator):
    """Generates SpinOpertor-like object

    Parameters
    ----------
    phases : float or array-like, shape (n_spins, )
        Phase(s) of pulse in **degrees**, i.e. 0 for +x, 90 for +y, 180 for -x,
        270 for -y.
    betas : float or array-like, shape (n_spins, )
        Flip angle(s) in **degrees**, i.e. 90 for pi/2 pulse and 180 for pi
        pulse
    active_spins : array-like of bool
        Spins which pulse acts on. Only required if beta is a float.
    spins : array-like, shape (n_spins, )
        Total spin quantum number of each spin centre. If None (default)
        then all spins are assumed to be S = 1/2.
    """

    def __init__(self, phases, betas, *, active_spins=None, spins=None):
        self.phases = phases
        self.betas = betas
        self.active_spins = active_spins
        self._parse_input()
        super().__init__(self._get_operators(spins))

    def _parse_input(self):
        """Check that enough information is given"""
        # If betas are iterable
        try:
            self.betas[0]
            self.betas_ = self.betas
            self.active_spins_ = np.logical_not(np.isclose(self.betas_, 0.0))
        # If only one flip angle given
        except TypeError:
            # If active spins also given
            if self.active_spins is not None:
                # Set all active pulses to have same flip angle
                self.betas_ = [self.betas if active else 0.0
                               for active in self.active_spins]
            # Otherwise complain
            else:
                mssg = 'If only one flip angle is specified, ' + \
                       'then active_spins must also be specified'
                raise ValueError(mssg)
        # Same trick for pulse phases
        try:
            self.phases[0]
            self.phases_ = self.phases
        except TypeError:
            # If active spins also given
            if self.active_spins is not None:
                # Set all active pulses to have the same phase
                self.phases_ = [self.phases if active else 0.0
                                for active in self.active_spins]
            # Otherwise complain
            else:
                mssg = 'If only one pulse phase is specified, ' + \
                       'then active_spins must also be specified'
                raise ValueError(mssg)

    def _get_operators(self, spins):
        op_list = []
        # Loop through spins
        n_spins = len(self.phases_)
        spins_inds = range(len(self.phases_))
        for i_spin, phase, beta in zip(spins_inds, self.phases_, self.betas_):
            # If this spin is active
            if ~np.isclose(beta, 0.0):
                # Get Cartesian components
                comps_x = Pulse.pulse_comps(n_spins, i_spin, 'x')
                comps_y = Pulse.pulse_comps(n_spins, i_spin, 'y')
                # Get powers of x and y pulses
                omega_1 = beta*np.pi/180.0
                coef_x = omega_1*np.cos(phase*np.pi/180.0)
                coef_y = omega_1*np.sin(phase*np.pi/180.0)
                # Make x-pulse
                if ~np.isclose(coef_x, 0.0):
                    op_list.append(ProductOperator(comps_x, coef=coef_x,
                                                   spins=spins))
                # Make y-pulse
                if ~np.isclose(coef_y, 0.0):
                    op_list.append(ProductOperator(comps_y, coef=coef_y,
                                                   spins=spins))
        # If no active pulses, use identity
        if op_list == []:
            identity_comps = ''.join(['i' for i_spin in range(n_spins)])
            op_list.append(ProductOperator(identity_comps, spins=spins))
        return op_list

    def from_spinop(op, *, spins=None):
        """Make Pulse from SpinOperator object

        Parameters
        ----------
        op : SpinOperator
            Operator representing Pulse
        spins : array-like, shape (n_spins, )
            Total spin quantum number of each spin centre. If None (default)
            then all spins are assumed to be S = 1/2.
        Returns
        -------
        Pulse
        """
        phases, betas = Pulse._decompose_spinop(op)
        return Pulse(phases, betas, spins)

    def from_ops(ops, *, spins=None):
        """Make Pulse from list of ProductOperators

        Parameters
        ----------
        ops : array-like of ProductOperator objects
            Operators which make up SpinOperator.
        spins : array-like, shape (n_spins, )
            Total spin quantum number of each spin centre. If None (default)
            then all spins are assumed to be S = 1/2.

        Returns
        -------
        Pulse
        """
        return Pulse.from_spinop(SpinOperator(ops), spins)

    def from_prodop(op, *, spins=None):
        """Make Pulse from a ProductOperator

        Parameters
        ----------
        op : ProductOperator objects
            Operator
        spins : array-like, shape (n_spins, )
            Total spin quantum number of each spin centre. If None (default)
            then all spins are assumed to be S = 1/2.

        Returns
        -------
        Pulse
        """
        return Pulse.from_ops([op], spins=None)

    @staticmethod
    def _decompose_spinop(op):
        # Get components of operator
        comps_list = op.get_comps_list()
        # Get number of spins
        n_spins = len(comps_list[0])
        # Make outputs
        phases = np.zeros((n_spins, ))
        betas = np.zeros((n_spins, ))
        # Loop through all spins
        for i_spin in range(n_spins):
            # Get components for pulsing this spin
            comps_x = Pulse.pulse_comps(n_spins, i_spin, 'x')
            comps_y = Pulse.pulse_comps(n_spins, i_spin, 'y')
            # Set coefficients to 0.0
            coef = 0.0
            # If x component present add power and delete from all operators
            if comps_x in comps_list:
                coef += op.get_coef(comps_x)
                comps_list.remove(comps_x)
            # If y component present add power and delete from all operators
            if comps_y in comps_list:
                coef += 1.0j*op.get_coef(comps_y)
                comps_list.remove(comps_y)
            # Get phase and power
            phases[i_spin] = np.angle(coef)*180.0/np.pi
            betas[i_spin] = np.abs(coef)*180.0/np.pi
        # Complain if pulses are left over
        if comps_list != []:
            mssg = 'Pulses cannot contain {} operators'
            raise ValueError(mssg.format(comps_list))
        return phases, betas

    def pulse_comps(n_spins, i_spin, comp):
        """Make Cartesian components string for a pulse

        Parameters
        ----------
        n_spins : int
            Length of comps
        i_spin : int
            Active spin
        comp : str
            Component

        Returns
        -------
        comps : str
            String of components
        """
        comps = ['i' for spin in range(n_spins)]
        comps[i_spin] = comp
        comps = ''.join(comp for comp in comps)
        return comps


class Observables(SpinOperator):
    """SpinOperator with all Cartesian component operators.

    As initialised, all operators have unit coefficients.

    Parameters
    ----------
    spins : array-like, shape (n_spins, )
        Total spin quantum number of each spin centre.
    identity : bool
        If True includes identity element.
    """

    def __init__(self, spins, *, identity=False):
        self.spins = spins
        self.identity = identity
        self._n_spins = len(spins)
        super().__init__(self._get_operators())

    def _get_operators(self):
        """Make list of all operators"""
        # Get all Cartesian components
        all_comps = _get_all_comps(self._n_spins)
        # Remove identity if requested
        if self.identity is False:
            all_comps.pop(0)
        # Return list of operators
        return [ProductOperator(comps, spins=self.spins)
                for comps in all_comps]


def ensure_pulse(op):
    """Convert PulseOperator, list of PulseOperator, or SpinOperator to Pulse
    """
    if isinstance(op, ProductOperator):
        return Pulse.from_prodop(op)
    elif isinstance(op, SpinOperator):
        return Pulse.from_spinop(op)
    elif isinstance(op, Pulse):
        return op
    else:
        try:
            op1 = isinstance(op[0], ProductOperator)
            if op1:
                return Pulse.from_ops(op)
        except TypeError:
            mssg = 'Could not convert operator to Pulse'
            raise ValueError(mssg)
    mssg = 'Could not convert operator to Pulse'
    raise ValueError(mssg)


if __name__ == '__main__':
    # Spin Hamiltonian frequencies
    wA = 0.5
    wB = -1.0
    wAB = 0.1
    # Pulses
    x_90 = Pulse(0, 90, active_spins=[True, True])
    x_180 = Pulse(0, 180, active_spins=[True, True])
    x_90 = Pulse(0, [90, 90], active_spins=[True, True])
    x_180 = Pulse([0, 90], 180, active_spins=[True, True])
    # Setup
    X = np.linspace(0, 20*np.pi, 101)
    sig = np.zeros((X.size, 4))
    # Thermal equilibrium
    rho0 = SpinOperator([ProductOperator('zi'), ProductOperator('iz')])
    # Apply first pulse
    rho0plus = rho0.nutate(x_90, force_propogator=True)
    # Loop through time delays
    for i in range(X.size - 1):
        # Get time
        t = X[i + 1]
        # Generate free-precession Hamiltonian
        tau = SpinOperator([ProductOperator('zi', coef=wA*t),
                            ProductOperator('iz', coef=wB*t),
                            ProductOperator('zz', coef=wAB*t)])
        # Free precession
        rhotauminus = rho0plus.nutate(tau, force_propogator=True)
        # Pi pulse
        rhotauplus = rhotauminus.nutate(x_180, force_propogator=True)
        # Refocuss
        rhotwotau = rhotauplus.nutate(tau, force_propogator=True)
        # Get expectation values
        sig[i, 0] = rhotwotau.expectation_value(ProductOperator('xz'))
        sig[i, 1] = rhotwotau.expectation_value(ProductOperator('yi'))
        sig[i, 2] = rhotwotau.expectation_value(ProductOperator('zx'))
        sig[i, 3] = rhotwotau.expectation_value(ProductOperator('iy'))

    # Plot expectation values
    fig, ax = plt.subplots()
    ax.plot(X, sig[:, 0], label='Sxz')
    ax.plot(X, sig[:, 1], label='Syi')
    ax.legend()
    fig, ax = plt.subplots()
    ax.plot(X, sig[:, 2], label='Szx')
    ax.plot(X, sig[:, 3], label='Siy')
    ax.legend()
    plt.show()
