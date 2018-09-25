from ..base import KernelBase,RegressorBase
from ..utils import check_dir,dump_pck,load_pck,dump_json,load_json,check_file
from copy import deepcopy
import importlib

class RegressorPipeline(RegressorBase):
    def __init__(self,steps=None,handler=None):
        if steps is None and handler is None:
            pass
        self.steps = steps
        self.params = []
        for name,step in self.steps: 
            self.params.append((name,step.get_params()))
            
        self.handler = handler
        self.X_train = None
    
    
    def fit(self,X,y):
        X,steps,global_data = self.handler.find_restart_checkpoint(X,self.steps)
        
        if 'X_train' in global_data:
            self.X_train = global_data['X_train']
        
        if len(steps) > 0: # if all the steps not already loaded
            for name,preprocessor in steps[:-1]:
                if isinstance(preprocessor,KernelBase):
                    ## if there is a kernel then keep a copy of 
                    ## the training samples for the predictions
                    self.X_train = deepcopy(X)
                X = preprocessor.fit(X).transform(X)
                self.handler.create_checkpoint(name,preprocessor,output=X)
                
            self._final_estimator.fit(X,y)
            self.handler.create_checkpoint(steps[-1][0],steps[-1][1])
        
    def predict(self,X):
        for name,preprocessor in self.steps[:-1]:
            #print name
            if isinstance(preprocessor,KernelBase):
                X = preprocessor.transform(X,X_train=self.X_train)
            else:
                X = preprocessor.transform(X)
        
        return self._final_estimator.predict(X)
    
    def get_params(self,deep=True):
        return self.params
    
    def get_summary(self,txt=False):
        summary = self._final_estimator.get_summary(txt)
        summary['parameters'] = self.params
        
        return summary

    @property
    def _final_estimator(self):
        return self.steps[-1][1]
    
    def dump(self):
        steps_pck = []
        
        for name,step in self.steps:
            cls_name = step.get_name()
            module_name = step.__module__
            steps_pck.append((name,cls_name,module_name,step.pack()))
        package = dict(X_train=self.X_train,steps_pck=steps_pck)
        return package

    def load(self,fn):
        
        data = load_pck(fn)
        self.handler = None
        
        self.params = []
        self.steps = []
        for name,cls_name,module_name,step_pack in data['steps_pck']: 
            my_module = importlib.import_module(module_name)
            MyStep = getattr(my_module, cls_name)
            step = MyStep()
            step.loads(step_pack)
            self.steps.append((name,step))
            self.params.append((name,step.get_params()))

        