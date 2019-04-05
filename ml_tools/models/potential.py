from ase.calculators.calculator import Calculator,all_changes
from copy import deepcopy

class MLCalculator(Calculator):
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
        self.model.representation.disable_pbar = True


    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        X = self.model.representation.transform([self.atoms])

        energy = self.model.predict(X,eval_gradient=False)
        forces = -self.model.predict(X,eval_gradient=True)

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces

