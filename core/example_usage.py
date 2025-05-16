"""Example usage of the permission management system"""
from permission_manager import PermissionManager, Permission, PermissionLevel

def main():
    # Initialize permission manager with config
    pm = PermissionManager("permissions_config.json")
    
    # Example: Check if StarCoder can modify code
    model_id = "starcoder"
    can_modify = pm.has_permission(model_id, Permission.MODIFY_EXISTING_LOGIC)
    print(f"Can StarCoder modify existing logic? {can_modify}")
    
    # Example: Validate file access
    file_path = "super_ai/core/self_modification.py"
    can_write = pm.validate_file_access(model_id, file_path, write_access=True)
    print(f"Can StarCoder write to {file_path}? {can_write}")
    
    # Example: Grant experimental features to base model
    base_model = "base_model"
    pm.grant_permission(base_model, Permission.USE_EXPERIMENTAL_FEATURES)
    print(f"Base model permissions: {pm.get_permissions(base_model)}")
    
    # Example: Set restricted level for a new model
    new_model = "safe_model"
    pm.set_permission_level(new_model, PermissionLevel.RESTRICTED)
    print(f"Safe model permissions: {pm.get_permissions(new_model)}")

if __name__ == "__main__":
    main()
