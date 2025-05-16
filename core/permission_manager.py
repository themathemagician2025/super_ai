"""
Permission Management System for Super AI
Handles granular access control for AI model operations
"""
from enum import Enum
from typing import Dict, Set, Optional
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    RESTRICTED = 0
    STANDARD = 1
    ELEVATED = 2
    ADMIN = 3

class Permission(Enum):
    READ_CODEBASE = "read_codebase"
    WRITE_CODE_MODULES = "write_code_modules"
    CREATE_NEW_SCRIPTS = "create_new_scripts"
    MODIFY_EXISTING_LOGIC = "modify_existing_logic"
    SELF_MODIFY = "self_modify"
    EXECUTE_PYTHON_CODE = "execute_python_code"
    ACCESS_LOGS_AND_METRICS = "access_logs_and_metrics"
    VERSION_CONTROL_ACCESS = "version_control_access"
    INTERNET_READ_ACCESS = "internet_read_access"
    PACKAGE_INSTALLATION = "package_installation"
    RUN_TESTS_AND_VALIDATE = "run_tests_and_validate"
    USE_LOCAL_LLM_APIS = "use_local_llm_apis"
    USE_EXPERIMENTAL_FEATURES = "use_experimental_features"
    REFACTOR_LARGE_CODE = "refactor_large_code"
    AUTOMATED_CRON_HOOKS = "automated_cron_hooks"
    TRACE_EXECUTION_FLOW = "trace_execution_flow"
    ACCESS_TRAINING_DATA = "access_training_data"
    STORE_CACHES_AND_MODELS = "store_caches_and_models"
    BUILD_UI_ELEMENTS = "build_ui_elements"
    CREATE_FALLBACK_LOGIC = "create_fallback_logic"

class PermissionManager:
    def __init__(self, config_path: Optional[str] = None):
        self.permissions: Dict[str, Set[Permission]] = {}
        self.permission_levels: Dict[str, PermissionLevel] = {}
        self._load_default_permissions()
        if config_path:
            self._load_config(config_path)

    def _load_default_permissions(self):
        """Initialize default permission sets for different levels"""
        # Restricted level permissions
        restricted = {
            Permission.READ_CODEBASE,
            Permission.ACCESS_LOGS_AND_METRICS,
            Permission.TRACE_EXECUTION_FLOW
        }

        # Standard level adds basic operational permissions
        standard = restricted | {
            Permission.CREATE_NEW_SCRIPTS,
            Permission.RUN_TESTS_AND_VALIDATE,
            Permission.BUILD_UI_ELEMENTS,
            Permission.ACCESS_TRAINING_DATA
        }

        # Elevated adds more powerful capabilities
        elevated = standard | {
            Permission.WRITE_CODE_MODULES,
            Permission.MODIFY_EXISTING_LOGIC,
            Permission.EXECUTE_PYTHON_CODE,
            Permission.VERSION_CONTROL_ACCESS,
            Permission.USE_LOCAL_LLM_APIS,
            Permission.CREATE_FALLBACK_LOGIC
        }

        # Admin has full access
        admin = set(Permission)

        self.level_permissions = {
            PermissionLevel.RESTRICTED: restricted,
            PermissionLevel.STANDARD: standard,
            PermissionLevel.ELEVATED: elevated,
            PermissionLevel.ADMIN: admin
        }

    def _load_config(self, config_path: str):
        """Load permission configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                for model_id, perms in config.get('permissions', {}).items():
                    self.permissions[model_id] = {Permission(p) for p in perms}
                for model_id, level in config.get('levels', {}).items():
                    self.permission_levels[model_id] = PermissionLevel(level)
        except Exception as e:
            logger.error(f"Failed to load permissions config: {e}")
            raise

    def set_permission_level(self, model_id: str, level: PermissionLevel):
        """Set the permission level for a model"""
        self.permission_levels[model_id] = level
        self.permissions[model_id] = self.level_permissions[level]

    def grant_permission(self, model_id: str, permission: Permission):
        """Grant a specific permission to a model"""
        if model_id not in self.permissions:
            self.permissions[model_id] = set()
        self.permissions[model_id].add(permission)

    def revoke_permission(self, model_id: str, permission: Permission):
        """Revoke a specific permission from a model"""
        if model_id in self.permissions:
            self.permissions[model_id].discard(permission)

    def has_permission(self, model_id: str, permission: Permission) -> bool:
        """Check if a model has a specific permission"""
        return (model_id in self.permissions and 
                permission in self.permissions[model_id])

    def get_permissions(self, model_id: str) -> Set[Permission]:
        """Get all permissions for a model"""
        return self.permissions.get(model_id, set())

    def validate_file_access(self, model_id: str, file_path: str, write_access: bool = False) -> bool:
        """Validate if a model can access a specific file"""
        if not self.has_permission(model_id, Permission.READ_CODEBASE):
            return False
        
        if write_access:
            if not self.has_permission(model_id, Permission.WRITE_CODE_MODULES):
                return False
                
            # Additional validation for sensitive paths
            sensitive_paths = ['core', 'security', 'self_modification']
            path = Path(file_path)
            if any(p in path.parts for p in sensitive_paths):
                return self.has_permission(model_id, Permission.SELF_MODIFY)
                
        return True

    def validate_code_execution(self, model_id: str, code_type: str) -> bool:
        """Validate if a model can execute specific types of code"""
        if code_type == 'python':
            return self.has_permission(model_id, Permission.EXECUTE_PYTHON_CODE)
        elif code_type == 'test':
            return self.has_permission(model_id, Permission.RUN_TESTS_AND_VALIDATE)
        elif code_type == 'experimental':
            return self.has_permission(model_id, Permission.USE_EXPERIMENTAL_FEATURES)
        return False

    def save_config(self, config_path: str):
        """Save current permission configuration to file"""
        config = {
            'permissions': {
                model_id: [p.value for p in perms]
                for model_id, perms in self.permissions.items()
            },
            'levels': {
                model_id: level.value
                for model_id, level in self.permission_levels.items()
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
