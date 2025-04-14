import os
import sys
from pathlib import Path

def setup_local_db():
    """Set up local SQLite database configuration"""
    # Find the config.py file
    config_path = Path("backend/app/core/config.py")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return False
        
    # Read the current content
    with open(config_path, "r") as f:
        content = f.read()
    
    # Check if we need to modify the file
    if "DATABASE_TYPE" not in content:
        print("Adding SQLite support to configuration...")
        
        # Find the right location to add DATABASE_TYPE
        settings_class_idx = content.find("class Settings(BaseSettings):")
        if settings_class_idx == -1:
            print("Error: Could not find Settings class definition")
            return False
            
        # Find the first field definition
        field_idx = content.find("Field(", settings_class_idx)
        if field_idx == -1:
            print("Error: Could not find Field definition")
            return False
            
        # Add DATABASE_TYPE before the first field
        insert_idx = content.rfind("\n", settings_class_idx, field_idx) + 1
        modified_content = (
            content[:insert_idx] + 
            "    DATABASE_TYPE: str = Field(default=\"sqlite\")  # \"postgresql\" or \"sqlite\"\n" + 
            content[insert_idx:]
        )
        
        # Find the __init__ method
        init_idx = modified_content.find("def __init__(self, **kwargs):")
        if init_idx == -1:
            # If __init__ doesn't exist, add it at the end of the class
            class_end_idx = modified_content.find("class ", settings_class_idx + 1)
            if class_end_idx == -1:
                class_end_idx = len(modified_content)
                
            # Find the last method in the class
            last_def_idx = modified_content.rfind("def ", settings_class_idx, class_end_idx)
            if last_def_idx == -1:
                print("Error: Could not determine where to add __init__ method")
                return False
                
            # Find the end of the last method
            last_method_end_idx = modified_content.find("\n\n", last_def_idx)
            if last_method_end_idx == -1:
                last_method_end_idx = class_end_idx
                
            # Add the __init__ method
            modified_content = (
                modified_content[:last_method_end_idx] + 
                "\n\n    def __init__(self, **kwargs):\n" +
                "        super().__init__(**kwargs)\n" +
                "        # Set database URI based on type\n" +
                "        if self.DATABASE_TYPE == \"sqlite\":\n" +
                "            self.SQLALCHEMY_DATABASE_URI = \"sqlite:///./app.db\"\n" +
                "        else:\n" +
                "            # Build PostgreSQL URI\n" +
                "            self.SQLALCHEMY_DATABASE_URI = (\n" +
                "                f\"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@\"\n" +
                "                f\"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}\"\n" +
                "            )\n" +
                modified_content[last_method_end_idx:]
            )
        else:
            # If __init__ exists, find where to add the database URI logic
            init_body_idx = modified_content.find("\n", init_idx) + 1
            if init_body_idx == 0:
                print("Error: Could not find init method body")
                return False
                
            # Find where to insert the SQLite logic
            if "SQLALCHEMY_DATABASE_URI" in modified_content[init_idx:]:
                # If SQLALCHEMY_DATABASE_URI is already set, replace it
                db_uri_idx = modified_content.find("self.SQLALCHEMY_DATABASE_URI", init_idx)
                if db_uri_idx != -1:
                    line_end_idx = modified_content.find("\n", db_uri_idx)
                    next_line_idx = modified_content.find("\n", line_end_idx + 1)
                    if line_end_idx == -1:
                        print("Error: Could not find end of SQLALCHEMY_DATABASE_URI line")
                        return False
                        
                    modified_content = (
                        modified_content[:db_uri_idx] + 
                        "        # Set database URI based on type\n" +
                        "        if self.DATABASE_TYPE == \"sqlite\":\n" +
                        "            self.SQLALCHEMY_DATABASE_URI = \"sqlite:///./app.db\"\n" +
                        "        else:\n" +
                        "            # Build PostgreSQL URI\n" +
                        "            self.SQLALCHEMY_DATABASE_URI = (\n" +
                        "                f\"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@\"\n" +
                        "                f\"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}\"\n" +
                        "            )" +
                        modified_content[next_line_idx:]
                    )
            else:
                # If SQLALCHEMY_DATABASE_URI is not set, add it
                modified_content = (
                    modified_content[:init_body_idx] + 
                    "        # Set database URI based on type\n" +
                    "        if self.DATABASE_TYPE == \"sqlite\":\n" +
                    "            self.SQLALCHEMY_DATABASE_URI = \"sqlite:///./app.db\"\n" +
                    "        else:\n" +
                    "            # Build PostgreSQL URI\n" +
                    "            self.SQLALCHEMY_DATABASE_URI = (\n" +
                    "                f\"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@\"\n" +
                    "                f\"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}\"\n" +
                    "            )\n" +
                    modified_content[init_body_idx:]
                )
        
        # Write the modified content back to the file
        with open(config_path, "w") as f:
            f.write(modified_content)
            
        print("✅ Added SQLite support to configuration")
        
        # Update .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r") as f:
                env_content = f.read()
                
            if "DATABASE_TYPE" not in env_content:
                with open(env_path, "a") as f:
                    f.write("\n# Use SQLite for local development\nDATABASE_TYPE=sqlite\n")
                print("✅ Updated .env with DATABASE_TYPE=sqlite")
        
        return True
    else:
        print("SQLite support already configured")
        return True

if __name__ == "__main__":
    setup_local_db()