import os

def fix_rawnet2_model():
    """Fix the RawNet2 model implementation"""
    rawnet2_path = "backend/app/models/audio_models.py"
    
    if not os.path.exists(rawnet2_path):
        print(f"Error: File not found - {rawnet2_path}")
        return False
        
    with open(rawnet2_path, "r") as f:
        content = f.read()
    
    # Check if RawNet2 model needs fixing
    if "self.skip = None" in content and "self.skip is not None" in content:
        print("Detected RawNet2 model implementation issue - fixing...")
        
        # Find the RawNet2 class definition
        start_idx = content.find("class RawNet2(nn.Module):")
        if start_idx == -1:
            print("Error: Could not find RawNet2 class definition")
            return False
            
        # Find the end of the class by looking for the next class definition
        next_class_idx = content.find("class ", start_idx + 1)
        if next_class_idx == -1:
            # If no next class, assume it extends to the end of the file
            rawnet2_code = content[start_idx:]
        else:
            rawnet2_code = content[start_idx:next_class_idx]
        
        # Replace with the fixed implementation
        fixed_rawnet2 = """class RawNet2(nn.Module):
    \"\"\"RawNet2 model for raw waveform analysis.\"\"\"
    
    def __init__(self, num_classes=1):
        super(RawNet2, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Residual blocks with properly initialized skip connections
        self.res_block1, self.skip1 = self._make_res_block(32, 32)
        self.res_block2, self.skip2 = self._make_res_block(32, 64, stride=2)
        self.res_block3, self.skip3 = self._make_res_block(64, 64)
        self.res_block4, self.skip4 = self._make_res_block(64, 128, stride=2)
        self.res_block5, self.skip5 = self._make_res_block(128, 128)
        self.res_block6, self.skip6 = self._make_res_block(128, 256, stride=2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        \"\"\"Create a residual block with proper skip connection\"\"\"
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        
        # Define skip connection
        skip = None
        if stride != 1 or in_channels != out_channels:
            skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers.append(nn.LeakyReLU(0.3))
        
        return nn.Sequential(*layers), skip
    
    def forward(self, x):
        # Ensure input is [B, 1, T] where B is batch size and T is time samples
        x = self.conv1(x)
        
        # Apply residual blocks with proper skip connections
        identity = x
        x = self.res_block1(x)
        if self.skip1 is not None:
            x = x + self.skip1(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block2(x)
        if self.skip2 is not None:
            x = x + self.skip2(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block3(x)
        if self.skip3 is not None:
            x = x + self.skip3(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block4(x)
        if self.skip4 is not None:
            x = x + self.skip4(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block5(x)
        if self.skip5 is not None:
            x = x + self.skip5(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block6(x)
        if self.skip6 is not None:
            x = x + self.skip6(identity)
        else:
            x = x + identity
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = F.leaky_relu(self.fc1(x), 0.3)
        x = self.fc2(x)
        
        return self.sigmoid(x)
        
    def get_features(self, x):
        \"\"\"Extract features before final classification.\"\"\"
        # Ensure input is [B, 1, T] where B is batch size and T is time samples
        x = self.conv1(x)
        
        # Apply residual blocks with proper skip connections
        identity = x
        x = self.res_block1(x)
        if self.skip1 is not None:
            x = x + self.skip1(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block2(x)
        if self.skip2 is not None:
            x = x + self.skip2(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block3(x)
        if self.skip3 is not None:
            x = x + self.skip3(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block4(x)
        if self.skip4 is not None:
            x = x + self.skip4(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block5(x)
        if self.skip5 is not None:
            x = x + self.skip5(identity)
        else:
            x = x + identity
        
        identity = x
        x = self.res_block6(x)
        if self.skip6 is not None:
            x = x + self.skip6(identity)
        else:
            x = x + identity
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        return x"""
        
        # Replace the RawNet2 class with the fixed version
        new_content = content[:start_idx] + fixed_rawnet2
        if next_class_idx != -1:
            new_content += content[next_class_idx:]
        
        # Write the fixed content back to the file
        with open(rawnet2_path, "w") as f:
            f.write(new_content)
            
        print("✅ Fixed RawNet2 model implementation")
        return True
    else:
        print("RawNet2 model seems to be already fixed or has a different implementation")
        return False

if __name__ == "__main__":
    fix_rawnet2_model()