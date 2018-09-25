from ..utils import check_dir,dump_pck,load_pck,dump_json,load_json,check_file
import numpy as np
from string import Template 
from time import ctime
import os.path as osp
from ..base import KernelBase
from glob import glob
from ..hasher.hasher import hash

class JsonMetadata(object):
    
    def __ini__(self):
        pass
    
    def set_data(self,parameter_hash,output_hash,state_fn,state_hash,step_name,
                base_hash,checksum,parameters,output_fn):
        self._data = dict(parameter_hash=parameter_hash,output_hash=output_hash,
                            state_fn=state_fn,state_hash=state_hash,
                            step_name=step_name,base_hash=base_hash,checksum=checksum,
                            parameters=parameters,output_fn=output_fn,timestamp=ctime())
    def get_data(self,):
        return self._data
    def __getitem__(self, key):
        return self._data[key]    
    def dump(self,fn):
        dump_json(fn,self._data)
    def load(self,fn):
        self._data = load_json(fn)

class HashJsonHandler(object):
    _metadata_prefix = 'metadata/'
    _metadata_extension = 'json'
    _data_prefix = 'data/'
    _data_extensions = {'npy':('npy',np.save,np.load),'pck':('pck',dump_pck,load_pck)}
    _metadata_fn_template = Template('${name}-${hash1}_${hash2}-${count}.${extension}')
    _data_fn_template = Template('${name}-${hash1}.${extension}')
     
    def __init__(self,path,restart=True,check_restart_file=True):
        self.check_restart_file = check_restart_file
        self.path = check_dir(path)
        self.restart = restart
        self.checksum = ''
        self.metadata_dir = check_dir(osp.join(self.path,self._metadata_prefix))
        self.data_dir = check_dir(osp.join(self.path,self._data_prefix))
    
    def reset(self):
        self.base_input_hash = None
        self.checksum = ''
        
    def set_base_input_hash(self,inp):
        self.base_input_hash = hash(inp)
        self.checksum += self.base_input_hash
        
    def check_where_to_restart(self,steps):
        restart_json_fns = []
        checksum = self.base_input_hash
        global_data_json_fns = {}
        ## try to find some metadata files to restart from
        for ii,(step_name,step) in enumerate(steps):
            params = step.get_params()
            
            if isinstance(step,KernelBase) is True: 
                # if this step is a kernel then the previous step has X_train. 
                # The kernel can't be first in the pipeline
                global_data_json_fns['X_train'] = ('output',restart_json_fns[-1])
                
            
            inp_param_hash = hash(params)
            checksum += inp_param_hash
            fns = glob(self.get_metadata_fn(step_name,inp_param_hash,all_possible=True))
            found = False
            if len(fns) > 0: # if there are some matching files, compare the checksum 
                for fn in fns:
                    metadata = JsonMetadata()
                    metadata.load(fn)
                    if metadata['checksum'] == checksum:
                        restart_json_fns.append(fn)
                        self.checksum = metadata['checksum']
                        found = True
                        break
                        
            if found is False: # matching files don't have the proper checksum
                if ii == 0: # There are no files to restart from
                    return None,None
                else: # at least one file to restart from
                    return restart_json_fns,global_data_json_fns
                                
        return restart_json_fns,global_data_json_fns
        
    def create_checkpoint(self,step_name,step,output=None):
        inp_param = step.get_params()
        inp_param_hash = hash(inp_param)
        base_hash = self.base_input_hash
        
        self.checksum += inp_param_hash
        
        ## Save the output
        output_fn = ''
        output_hash = hash(output)
        if output is not None: # last estimator does not have output
            output_fn,dump_func,_ = self.get_output_fn(step_name,output_hash,output)
            dump_func(output_fn,output)
            
        ## Save the state of the step
        state = step.pack()
        state_hash = hash(state)
        state_fn,dump_func,_ = self.get_state_fn(step_name,state_hash)
        dump_func(state_fn,state)
        
        ## Save Metadata
        metadata_fn = self.get_metadata_fn(step_name,inp_param_hash)
        metadata = JsonMetadata()
        metadata.set_data(**dict(parameter_hash=inp_param_hash,output_hash=output_hash,
                        state_fn=state_fn,state_hash=state_hash,
                        step_name=step_name,base_hash=base_hash,checksum=self.checksum,
                        parameters=inp_param,output_fn=output_fn))
        
        metadata.dump(metadata_fn)
            
    def find_restart_checkpoint(self,X,steps):
        
        self.set_base_input_hash(X)
        
        global_data = {}
        if self.restart is False: # start from the begining 
            return X,steps,global_data
        
        steps_params = []
        for name,step in steps: 
            steps_params.append((name,step.get_params()))
        
        restart_json_fns,global_data_json_fns = self.check_where_to_restart(steps)
        
        if restart_json_fns is None: # there are no files to restart from
            return X,steps,global_data
        
        ## restart from the step after the last checkpoint
        restart_idx = len(restart_json_fns)
        
        ## load the state of the estimators
        for ii,fn in enumerate(restart_json_fns):
            state_package = self.get_data(fn,fieldname='state')
            steps[ii][-1].unpack(state_package)
        
        ## load the output of the last checkpoint
        if restart_idx < len(steps): 
            # if the last checkpoint is after the model fit then there is no output
            fn = restart_json_fns[-1]
            previous_output = self.get_data(fn,fieldname='output')
        else:
            previous_output = None
        
        for k,(fieldname,fn) in global_data_json_fns.iteritems():
            global_data[k] = self.get_data(fn,fieldname=fieldname)
        
        
        return previous_output,steps[restart_idx:],global_data
    
    def get_data(self,metadata_fn,fieldname='output'):
        
        metadata = load_json(metadata_fn)
        
        data_fn = metadata[fieldname+'_fn']
        ext = data_fn.split('.')[-1]
        _, _, load_func = self._data_extensions[ext]
        
        data = load_func(data_fn)
        
        if self.check_restart_file is True:
            error_m = "Loaded data {}@{} does not match the metadata hash {}".format(
                fieldname,metadata[fieldname+'_fn'],metadata_fn)
            assert hash(data) == metadata[fieldname+'_hash'], error_m
                
        return data
    
    def get_metadata_fn(self,step_name,inp_param_hash,all_possible=False):
        """ 
        if all_possible==True return a filename with * instead of index to glob the metadatafilenames compatible
        with this particular inputs and structure 
        else give a filename that does not exist already
        """
        base_hash = self.base_input_hash
    
        fn = self._metadata_fn_template.safe_substitute(
                dict(name=step_name,hash1=base_hash,hash2=inp_param_hash,extension=self._metadata_extension)
                )
        if all_possible is False:
            metadata_fn = self._get_unique_fn(osp.join(self.metadata_dir,fn))
        else:
            metadata_fn = Template(osp.join(self.metadata_dir,fn)).safe_substitute(count='*')
            
        return metadata_fn
    
    def get_state_fn(self,step_name,state_hash):
        
        data_extension,dump_func,load_func = self._data_extensions['pck']
        ## Assumes that the hash is unique (no collision) here so no possible overwrite of different data
        fn = self._data_fn_template.safe_substitute(
                dict(name=step_name,hash1=state_hash,extension=data_extension)
                )
        state_fn = osp.join(self.data_dir,fn)
        
        return state_fn,dump_func,load_func
    
    def get_output_fn(self,step_name,output_hash,output):
        
        if isinstance(output, np.ndarray):
            data_extension,dump_func,load_func = self._data_extensions['npy']
        elif isinstance(output, dict):
            data_extension,dump_func,load_func = self._data_extensions['pck']
        else:
            raise ValueError('Output type {} is not handled'.format(type(output)))
        ## Assumes that the hash is unique (no collision) here so no possible overwrite of different data
        fn = self._data_fn_template.safe_substitute(
                dict(name=step_name,hash1=output_hash,extension=data_extension)
                )
        output_fn = osp.join(self.data_dir,fn)
        
        return output_fn,dump_func,load_func

    def _get_unique_fn(self,fn_temp):
        count = 0
        aa = True
        while aa:
            if  check_file(Template(fn_temp).safe_substitute(count=count)) is False:
                aa = False
            else:
                count += 1
        return Template(fn_temp).safe_substitute(count=count)
    
    def get_name(self):
        return type(self).__name__
