from __future__ import annotations
import functools
import logging
import os
from typing import TYPE_CHECKING, Sequence


import numpy as np
import numpy.typing as npt
from scipy.stats import unitary_group
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.location import CircuitLocationLike
from bqskit.ir.location import CircuitLocation
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskitgpu.unitary_acc import VariableUnitaryGateAcc
from bqskitgpu.unitarybuilderjax import UnitaryBuilderJax
from bqskitgpu.unitarymatrixjax import UnitaryMatrixJax
from bqskitgpu.singlelegedtensor import SingleLegSideTensor, RHSTensor, LHSTensor

from bqskit.ir.gates import  CXGate

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.state.state import StateLike
    from bqskit.qis.state.system import StateSystemLike
    from bqskit.qis.unitary.unitarymatrix import UnitaryLike

_logger = logging.getLogger(__name__)

jax.config.update('jax_enable_x64', True)



class QFactor_sample_jax(Instantiater):


    def __init__(
            self,
            dist_tol: float = 1e-10,
            max_iters: int = 100000,
            min_iters: int = 1000,
            beta: float = 0.0,
            amount_of_validation_states : int = 2,
        ):


        if not isinstance(dist_tol, float) or dist_tol > 0.5:
            raise TypeError('Invalid distance threshold.')

        if not isinstance(max_iters, int) or max_iters < 0:
            raise TypeError('Invalid maximum number of iterations.')

        if not isinstance(min_iters, int) or min_iters < 0:
            raise TypeError('Invalid minimum number of iterations.')


        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters

        self.beta = beta
        self.amount_of_validation_states = amount_of_validation_states


    def instantiate(
            self,
            circuit: Circuit,
            target: UnitaryMatrix | StateVector | StateSystem,
            x0: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:

        return self.multi_start_instantiate(circuit, target, 1)

    def multi_start_instantiate_inplace(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> None:
        """
        Instantiate `circuit` to best implement `target` with multiple starts.

        See :func:`multi_start_instantiate` for more info.

        Notes:
            This method is a version of :func:`multi_start_instantiate`
            that modifies `circuit` in place rather than returning a copy.
        """
        target = self.check_target(target)
        params = self.multi_start_instantiate(circuit, target, num_starts)
        circuit.set_params(params)


    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:
        
        return self.multi_start_instantiate(circuit, target, num_starts)
    



    def multi_start_instantiate(
        self,
        circuit: Circuit,
        target: UnitaryLike | StateLike | StateSystemLike,
        num_starts: int,
    ) -> npt.NDArray[np.float64]:
        if len(circuit) == 0:
            return []

        circuit = circuit.copy()

        # A very ugly casting
        for op in circuit:
            g = op.gate
            if isinstance(g, VariableUnitaryGate):
                g.__class__ = VariableUnitaryGateAcc


        target = UnitaryMatrixJax(self.check_target(target))
        locations = tuple([op.location for op in circuit])
        gates = tuple([op.gate for op in circuit])
        untrys = []

        for gate in gates:
            size_of_untry = 2**gate.num_qudits

            if isinstance(gate, VariableUnitaryGateAcc):
                pre_padding_untrys = [
                    unitary_group.rvs(size_of_untry) for
                    _ in range(num_starts)
                ]
            else:
                pre_padding_untrys = [
                    gate.get_unitary().numpy for
                    _ in range(num_starts)
                ]

            untrys.append(pre_padding_untrys)

        untrys = jnp.array(np.stack(untrys, axis=1))



    def state_sample_sweep(N:int, target:UnitaryMatrixJax, locations, gates, untrys, num_qudits, radixes,  dist_tol, max_iters:int = 1000, beta:float=0.0, training_states_kets = None, validation_states_kets = None):


        training_costs = []
        validation_costs = []

        amount_of_gates = len(gates)

        training_states_bras = []
        if training_states_kets == None:
            training_states_kets = QFactor_sample_jax.generate_random_states(N, np.prod(radixes))
        else:
            assert N==len(training_states_kets)

        for ket in training_states_kets:
            training_states_bras.append(ket.T.conj())

        validation_states_bras = []
        if validation_states_kets == None:
            validation_states_kets = QFactor_sample_jax.generate_random_states(N//4, np.prod(radixes))    

        for ket in validation_states_kets:
            validation_states_bras.append(ket.T.conj())


        

        #### Compute As
        target_dagger = target.T.conj()
        A_train = RHSTensor(list_of_states = training_states_bras, num_qudits=num_qudits, radixes=radixes)
        A_train.apply_left(target_dagger, range(num_qudits))

        A_val = RHSTensor(list_of_states = validation_states_bras, num_qudits=num_qudits, radixes=radixes)
        A_val.apply_left(target_dagger, range(num_qudits))

        B0_train = LHSTensor(list_of_states = training_states_kets, num_qudits=num_qudits, radixes=radixes)
        B0_val = LHSTensor(list_of_states = validation_states_kets, num_qudits=num_qudits, radixes=radixes)

        ### Until done....
        it = 1
        while True:

            ### Compute B
            
            B = [B0_train]
            for location, utry in zip(locations[:-1], untrys[:-1]):
                B.append(B[-1].copy())
                B[-1].apply_right(utry, location)

            temp = B[-1].copy()
            temp.apply_right(untrys[-1], locations[-1])
            training_cost = 2*(1-jnp.real(SingleLegSideTensor.calc_env(temp, A_train, [])[0])/N)
            # print(f'initial {cost = }')

            ### iterate over every gate from right to left and update it
            new_untrys = [None]*amount_of_gates
            a_train:RHSTensor = A_train.copy()
            a_val:RHSTensor = A_val.copy()
            for idx in reversed(range(amount_of_gates)):
                b = B[idx]
                gate = gates[idx]
                location = locations[idx]
                utry = untrys[idx]
                if gate.num_params > 0:
                    # print(f"Updating {utry._utry = }")
                    env = SingleLegSideTensor.calc_env(b, a_train, location)
                    utry = gate.optimize(env.T, get_untry=True, prev_utry=utry, beta=beta)
                    # print(f"to {utry._utry = }")
                
                new_untrys[idx] = utry
                a_train.apply_left(utry, location)
                a_val.apply_left(utry, location)

            untrys = new_untrys


            training_cost = 2*(1-jnp.real(SingleLegSideTensor.calc_env(B0_train, a_train, [])[0])/N)
            validation_cost = 2*(1-jnp.real(SingleLegSideTensor.calc_env(B0_val, a_val, [])[0])/(N//4))

            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
            
            if it%100 == 0:
                print(f'{it = } {training_cost = }')
                print(f'{it = } {validation_cost = }')

            if it > 10 and validation_cost/32 > training_cost:
                print("Stopped due to no improvment in validation set")
                break
            if training_cost <= dist_tol or it > 3000:
                print(f'{it = } {training_cost = }')
                break
            it+=1



        plt.figure(figsize=(10, 6))
        plt.plot(training_costs, label='Training Costs', marker='o')
        plt.plot(validation_costs, label='Validation Costs', marker='x')

        # Add labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.title('Training and Validation Costs')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()

        params = []
        for untry, gate in zip(untrys, gates):
            if  isinstance(gate, ConstantGate):
                params.extend([])
            else:    
                params.extend(
                    gate.get_params(untry.numpy),
            )
            
        return np.array(params)





    @staticmethod
    def generate_random_states(amount_of_states, size_of_state):
        """
        Generate a list of random state vectors (kets) using random unitary matrices.

        This function generates a specified number of random quantum state vectors
        (kets) by creating random unitary matrices and extracting their first columns.

        Args:
            amount_of_states (int): The number of random states to generate.
            size_of_state (int): The dimension of each state vector (ket).

        Returns:
            list of ndarrays: A list containing random quantum state vectors (kets).
                            Each ket is represented as a numpy ndarray of shape (size_of_state, 1).
        """
        states_kets = []
        states_to_add = amount_of_states
        while states_to_add > 0:
            # We generate a random unitary and take its columns
            rand_unitary = unitary_group.rvs(size_of_state)
            states_to_add_in_step = min(states_to_add, size_of_state)
            for i in range(states_to_add_in_step):
                states_kets.append(rand_unitary[:, i:i+1])
            states_to_add -= states_to_add_in_step
        return states_kets