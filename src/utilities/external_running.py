import os
import subprocess
import shutil
from src.utilities.definitions import *

class ExternalToolRunner:
    def __init__(self):
        self.ants_path = ANTS_PATH
        self.fsl_path = FSL_PATH
        self.niftyreg_path = NIFTYREG_PATH
        self.afni_path = AFNI_PATH
        
    def _add_tool_path_to_env(self, tool_bin_dir, env):
        if tool_bin_dir:
            env['PATH'] = tool_bin_dir + os.pathsep + env['PATH']
        return env
    
    def run_command(self, command, tool='ants'):
        env = os.environ.copy()
        
        if tool == 'ants' and self.ants_path:
            env = self._add_tool_path_to_env(self.ants_path, env)
        elif tool == 'fsl' and self.fsl_path:
            env = self._add_tool_path_to_env(self.fsl_path, env)
        elif tool == 'niftyreg' and self.niftyreg_path:
            env = self._add_tool_path_to_env(self.niftyreg_path, env)
        elif tool == 'afni' and self.afni_path:
            env = self._add_tool_path_to_env(self.afni_path, env)
            
        
        command_name = command.split()[0]
        if not shutil.which(command_name, path=env['PATH']):
            raise EnvironmentError(
                f"{command_name} not found in PATH. Make sure {tool.upper()} is installed "
                "and the binary directory is added to your PATH in definition.py."
            )
        
        subprocess.run(command, shell=True, env=env, check=True)