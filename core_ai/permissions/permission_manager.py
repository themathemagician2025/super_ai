import os
import subprocess
import sys
from pathlib import Path
from typing import Set, Dict, List, Optional
import git
import logging
import json
from enum import Enum, auto

class Permission(Enum):
    READ_CODEBASE = auto()
    WRITE_CODE_MODULES = auto()
    CREATE_NEW_SCRIPTS = auto()
    MODIFY_EXISTING_LOGIC = auto()
    SELF_MODIFY = auto()
    EXECUTE_PYTHON_CODE = auto()
    ACCESS_LOGS_METRICS = auto()
    VERSION_CONTROL_ACCESS = auto()
    INTERNET_READ_ACCESS = auto()
    PACKAGE_INSTALLATION = auto()
    RUN_TESTS_VALIDATE = auto()
    USE_LOCAL_LLM_APIS = auto()
    USE_EXPERIMENTAL = auto()
    REFACTOR_LARGE_CODE = auto()
    AUTOMATED_CRON_HOOKS = auto()
    TRACE_EXECUTION_FLOW = auto()
    ACCESS_TRAINING_DATA = auto()
    STORE_CACHES_MODELS = auto()
    BUILD_UI_ELEMENTS = auto()
    CREATE_FALLBACK_LOGIC = auto()

class PermissionManager:
    def __init__(self, workspace_root: Path):
        self.workspace = workspace_root
        self.repo = git.Repo(workspace_root)
        self.permissions: Set[Permission] = set()
        self.permission_hooks: Dict[Permission, List[callable]] = {}
        self._init_permission_hooks()
        
    def _init_permission_hooks(self):
        self.permission_hooks = {
            Permission.READ_CODEBASE: [self._validate_file_read],
            Permission.WRITE_CODE_MODULES: [self._validate_file_write],
            Permission.CREATE_NEW_SCRIPTS: [self._validate_script_creation],
            Permission.MODIFY_EXISTING_LOGIC: [self._validate_logic_modification],
            Permission.SELF_MODIFY: [self._validate_self_modification],
            Permission.EXECUTE_PYTHON_CODE: [self._validate_code_execution],
            Permission.ACCESS_LOGS_METRICS: [self._validate_logs_access],
            Permission.VERSION_CONTROL_ACCESS: [self._validate_git_access],
            Permission.INTERNET_READ_ACCESS: [self._validate_internet_access],
            Permission.PACKAGE_INSTALLATION: [self._validate_package_install],
            Permission.RUN_TESTS_VALIDATE: [self._validate_test_execution],
            Permission.USE_LOCAL_LLM_APIS: [self._validate_llm_access],
            Permission.USE_EXPERIMENTAL: [self._validate_experimental],
            Permission.REFACTOR_LARGE_CODE: [self._validate_refactoring],
            Permission.AUTOMATED_CRON_HOOKS: [self._validate_cron_access],
            Permission.TRACE_EXECUTION_FLOW: [self._validate_trace_access],
            Permission.ACCESS_TRAINING_DATA: [self._validate_data_access],
            Permission.STORE_CACHES_MODELS: [self._validate_storage_access],
            Permission.BUILD_UI_ELEMENTS: [self._validate_ui_access],
            Permission.CREATE_FALLBACK_LOGIC: [self._validate_fallback_creation]
        }

    def grant_permission(self, permission: Permission):
        self.permissions.add(permission)

    def revoke_permission(self, permission: Permission):
        self.permissions.discard(permission)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def validate_operation(self, permission: Permission, **kwargs) -> bool:
        if not self.has_permission(permission):
            return False
        hooks = self.permission_hooks.get(permission, [])
        return all(hook(**kwargs) for hook in hooks)

    def _validate_file_read(self, file_path: Path = None, **kwargs) -> bool:
        if not file_path:
            return False
        allowed_extensions = {'.py', '.js', '.json', '.yaml', '.yml', '.cfg', '.ini'}
        return file_path.suffix.lower() in allowed_extensions

    def _validate_file_write(self, file_path: Path = None, **kwargs) -> bool:
        if not file_path:
            return False
        allowed_dirs = {'src', 'modules', 'agents', 'core', 'utils'}
        return any(part in allowed_dirs for part in file_path.parts)

    def _validate_script_creation(self, file_path: Path = None, **kwargs) -> bool:
        return self._validate_file_write(file_path)

    def _validate_logic_modification(self, file_path: Path = None, **kwargs) -> bool:
        if not file_path:
            return False
        core_modules = {'prediction', 'analysis', 'fallback', 'core'}
        return any(module in file_path.parts for module in core_modules)

    def _validate_self_modification(self, file_path: Path = None, **kwargs) -> bool:
        if not file_path:
            return False
        self_mod_files = {'self_modification.py', 'code_generator.py', 'evolution_manager.py'}
        return file_path.name in self_mod_files

    def _validate_code_execution(self, code: str = None, **kwargs) -> bool:
        if not code:
            return False
        unsafe_patterns = {'os.system', 'subprocess.', 'eval(', 'exec('}
        return not any(pattern in code for pattern in unsafe_patterns)

    def _validate_logs_access(self, log_path: Path = None, **kwargs) -> bool:
        if not log_path:
            return False
        return 'logs' in log_path.parts or 'metrics' in log_path.parts

    def _validate_git_access(self, operation: str = None, **kwargs) -> bool:
        allowed_ops = {'commit', 'revert', 'diff', 'log', 'branch'}
        return operation in allowed_ops

    def _validate_internet_access(self, url: str = None, **kwargs) -> bool:
        if not url:
            return False
        allowed_domains = {'python.org', 'pypi.org', 'github.com', 'githubusercontent.com'}
        return any(domain in url for domain in allowed_domains)

    def _validate_package_install(self, package: str = None, **kwargs) -> bool:
        if not package:
            return False
        with open(self.workspace / 'requirements.txt') as f:
            allowed_packages = {line.split('==')[0].strip() for line in f}
        return package in allowed_packages

    def _validate_test_execution(self, test_path: Path = None, **kwargs) -> bool:
        if not test_path:
            return False
        return 'tests' in test_path.parts

    def _validate_llm_access(self, model: str = None, **kwargs) -> bool:
        allowed_models = {'starcoder', 'phi', 'dolphin', 'llama'}
        return model.lower() in allowed_models

    def _validate_experimental(self, feature: str = None, **kwargs) -> bool:
        allowed_features = {'speculative_generation', 'prompt_chaining', 'meta_coding'}
        return feature in allowed_features

    def _validate_refactoring(self, files: List[Path] = None, **kwargs) -> bool:
        if not files:
            return False
        return len(files) <= 10  # Limit number of files to refactor at once

    def _validate_cron_access(self, schedule: str = None, **kwargs) -> bool:
        if not schedule:
            return False
        return True  # Add specific schedule validation if needed

    def _validate_trace_access(self, trace_type: str = None, **kwargs) -> bool:
        allowed_traces = {'stack', 'memory', 'error'}
        return trace_type in allowed_traces

    def _validate_data_access(self, data_path: Path = None, **kwargs) -> bool:
        if not data_path:
            return False
        allowed_dirs = {'data', 'datasets', 'training'}
        return any(d in data_path.parts for d in allowed_dirs)

    def _validate_storage_access(self, storage_type: str = None, **kwargs) -> bool:
        allowed_storage = {'model_weights', 'fine_tuned', 'prompt_cache', 'response_cache'}
        return storage_type in allowed_storage

    def _validate_ui_access(self, ui_type: str = None, **kwargs) -> bool:
        allowed_ui = {'dashboard', 'viewer', 'predictor', 'analyzer'}
        return ui_type in allowed_ui

    def _validate_fallback_creation(self, module: str = None, **kwargs) -> bool:
        allowed_modules = {'prediction', 'analysis', 'processing', 'core'}
        return module in allowed_modules

    def get_permission_status(self) -> Dict[str, bool]:
        return {perm.name: perm in self.permissions for perm in Permission}

    def save_permissions(self, file_path: Path):
        with open(file_path, 'w') as f:
            json.dump({
                'permissions': [p.name for p in self.permissions]
            }, f, indent=2)

    def load_permissions(self, file_path: Path):
        with open(file_path) as f:
            data = json.load(f)
            self.permissions = {Permission[p] for p in data['permissions']} 