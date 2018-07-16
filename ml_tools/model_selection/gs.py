from  itertools import product
from ..utils import tqdm_cs
from ..models.pipelines import RegressorPipeline

class StepsGrid(object):
    def __init__(self,steps):
        """ parameters is a list of tuples (name, parameters_dictionary)""" 
        parameters_fixed = {k:v['fixed'] for k,obj,v in steps}
        parameters_gs = {k:v['gs'] for k,obj,v in steps}
        # reversed so that the __iter__ give parameters that change primerly on the 1st step of the pipeline
        self.parameters_gs_names_sorted = [(k,sorted([name for name in v['gs']]))  for k,obj,v in reversed(steps)]
        self.step_names =  [k for k,obj,v in steps]
        self.step_obj = {k:obj for k,obj,v in steps}
        self.N_iter = 1
        self.parameters_gs = {}
        self.param_names = {k:sorted([name for name in v])for k,v in parameters_gs.iteritems() }
        
        self.parameters_to_display = {'low':   {k:[]  for k,v in parameters_fixed.iteritems()},
                                      'medium':{k:[]  for k,v in parameters_fixed.iteritems()},
                                      'high':  {k:[name for name in v]  for k,v in parameters_fixed.iteritems() }}
        for name in self.step_names:
            for k,v in parameters_gs[name].iteritems():
                if is_iterable(v):
                    self.N_iter *= len(v)
                    self.parameters_gs[k] = v
                    self.parameters_to_display['low'][name].append(k)
                else:
                    self.parameters_gs[k] = [v]
                self.parameters_to_display['medium'][name].append(k)
                self.parameters_to_display['high'][name].append(k)
                
        
        self.out = [( step_name,dict(obj=None,params={k:None for k in parameters_fixed[step_name].keys()+parameters_gs[step_name].keys()}) )
                                                                                for step_name in self.step_names]           
        for step_name,step in self.out:
            step['obj'] = self.step_obj[step_name]
            for k in parameters_fixed[step_name]:
                step['params'][k] = parameters_fixed[step_name][k]
        
        
    def __iter__(self):
        # self.parameters_gs_names_sorted makes sure that items is sorted by step_name / name
        # so that i/o collisions on the 1st step of the pipeline are minimized 
        items = [(name,self.parameters_gs[name]) for k,names in self.parameters_gs_names_sorted for name in names]
        keys, values = zip(*items)
        for ii in product(*values):
            params = dict(zip(keys, ii))
            for step_name,step in self.out:
                for k in self.param_names[step_name]:
                    step['params'][k] = params[k]
            yield self.out
            
    def __len__(self):
        return self.N_iter
        
def is_iterable(obj):
    from collections import Iterable
    return isinstance(obj, Iterable)

  
class GridSearch(object):
    def __init__(self,steps_grid,handler,disable_pbar=False,verbosity='low'):
        self.handler = handler
        self.steps_grid = StepsGrid(steps_grid)
        self.disable_pbar = disable_pbar
        self.verbosity = verbosity if verbosity in ['low','medium','high'] else 'low'
    
    def fit(self,X,y):
        self.results = []
        for steps in tqdm_cs(self.steps_grid,disable=self.disable_pbar):
            #for name,step in steps
            steps_init = [(name,step['obj'](**step['params'])) for name,step in steps]
            self.handler.reset()
            model = RegressorPipeline(steps_init,self.handler)
            model.fit(X,y)
            
            summary = model.get_summary()
            aa = {k:v for k,v in summary['score'].iteritems()}
            if self.verbosity in ['medium','high']:
                aa.update(**{'predictions':summary['predictions']})
            bb = {}
            for name,step in steps:
                for k in self.steps_grid.parameters_to_display[self.verbosity][name]:
                    if isinstance(step['params'][k],dict):
                        for kk,v in step['params'][k].iteritems():
                            bb[kk] = v
                    else:
                        bb[k] = step['params'][k] 
            aa.update(**bb)
            self.results.append(aa)
            
    def get_summary(self):
        return self.results