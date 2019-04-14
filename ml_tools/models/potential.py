from ase.calculators.calculator import Calculator,all_changes
from copy import deepcopy
from ..base import BaseIO

class MLCalculator(BaseIO,Calculator):
    """
    """

    implemented_properties = ['energy','forces']
    'Properties calculator can handle (energy, forces, ...)'

    default_parameters = {}
    'Default parameters'

    nolabel = True

    def __init__(self, model, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model = deepcopy(model)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        X = [self.atoms]
        for v in self.model.feature_transformations:
            X = v.transform(X)
        
        energy = self.model.predict(X,eval_gradient=False)
        forces = -self.model.predict(X,eval_gradient=True)

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces

