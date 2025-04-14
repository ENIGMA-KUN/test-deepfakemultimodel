# RawNet2 Model Implementation Fix Guide

## Problem Description

After carefully analyzing the RawNet2 model implementation in `backend/app/models/audio_models.py`, we identified a critical issue in the forward pass method. The issue revolves around how the residual skip connections are handled.

### Current Implementation (Problematic)

```python
def forward(self, x):
    # Ensure input is [B, 1, T] where B is batch size and T is time samples
    x = self.conv1(x)
    
    # Apply residual blocks
    if self.skip is not None:
        x = self.res_block1(x) + self.skip(x)
    else:
        x = self.res_block1(x) + x
    
    self.skip = None
    x = self.res_block2(x)
    
    self.skip = None
    x = self.res_block3(x) + x
    
    self.skip = None
    x = self.res_block4(x)
    
    self.skip = None
    x = self.res_block5(x) + x
    
    self.skip = None
    x = self.res_block6(x)
    
    # Global pooling
    x = self.global_pool(x).squeeze(-1)
    
    # Classification
    x = F.leaky_relu(self.fc1(x), 0.3)
    x = self.fc2(x)
    
    return self.sigmoid(x)
```

### Issues:

1. **Class Attribute Misuse**: `self.skip` is being used as if it's a class attribute, but it's only conditionally defined in the `_make_res_block` method. This leads to unpredictable behavior when checking `if self.skip is not None` before it's properly initialized.

2. **Skip Connection Logic**: The skip connections for the residual blocks are being manually set to `None` after each block, but this approach is error-prone and can lead to unpredictable behavior.

3. **Code Structure**: For each residual block, the check for whether to apply the skip connection is not consistent. Some blocks add the input (`+ x`) regardless of the skip condition, while others don't.

## Proposed Solution

### 1. Refactor the RawNet2 Class Definition

Each residual block should manage its own skip connection internally rather than relying on a class-level `self.skip` attribute. We'll modify the `_make_res_block` method to return both the main path and a skip connection function.

### 2. Fix the Forward Method

Rewrite the forward method to properly handle each residual block's skip connection.

## Implementation

Here's the corrected implementation:

```python
class RawNet2(nn.Module):
    """RawNet2 model for raw waveform analysis."""
    
    def __init__(self, num_classes=1):
        super(RawNet2, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 32)
        self.res_block2 = self._make_res_block(32, 64, stride=2)
        self.res_block3 = self._make_res_block(64, 64)
        self.res_block4 = self._make_res_block(64, 128, stride=2)
        self.res_block5 = self._make_res_block(128, 128)
        self.res_block6 = self._make_res_block(128, 256, stride=2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
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
        main1, skip1 = self.res_block1
        if skip1 is not None:
            x = main1(x) + skip1(x)
        else:
            x = main1(x) + x
            
        main2, skip2 = self.res_block2
        if skip2 is not None:
            x = main2(x) + skip2(x)
        else:
            x = main2(x) + x
            
        main3, skip3 = self.res_block3
        if skip3 is not None:
            x = main3(x) + skip3(x)
        else:
            x = main3(x) + x
            
        main4, skip4 = self.res_block4
        if skip4 is not None:
            x = main4(x) + skip4(x)
        else:
            x = main4(x) + x
            
        main5, skip5 = self.res_block5
        if skip5 is not None:
            x = main5(x) + skip5(x)
        else:
            x = main5(x) + x
            
        main6, skip6 = self.res_block6
        if skip6 is not None:
            x = main6(x) + skip6(x)
        else:
            x = main6(x) + x
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = F.leaky_relu(self.fc1(x), 0.3)
        x = self.fc2(x)
        
        return self.sigmoid(x)
```

## Alternative Implementation

If the above implementation is too drastically different from the original code structure, here's an alternative that's closer to the original but still fixes the issue:

```python
class RawNet2(nn.Module):
    """RawNet2 model for raw waveform analysis."""
    
    def __init__(self, num_classes=1):
        super(RawNet2, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 32)
        self.res_block2 = self._make_res_block(32, 64, stride=2)
        self.res_block3 = self._make_res_block(64, 64)
        self.res_block4 = self._make_res_block(64, 128, stride=2)
        self.res_block5 = self._make_res_block(128, 128)
        self.res_block6 = self._make_res_block(128, 256, stride=2)
        
        # Skip connections for each block
        self.skip1 = self._make_skip_connection(32, 32, 1)
        self.skip2 = self._make_skip_connection(32, 64, 2)
        self.skip3 = self._make_skip_connection(64, 64, 1)
        self.skip4 = self._make_skip_connection(64, 128, 2)
        self.skip5 = self._make_skip_connection(128, 128, 1)
        self.skip6 = self._make_skip_connection(128, 256, 2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.3))
        
        return nn.Sequential(*layers)
    
    def _make_skip_connection(self, in_channels, out_channels, stride=1):
        if stride != 1 or in_channels != out_channels:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        return None
    
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
```

## Implementation Notes

1. **Memory Usage**: The second implementation might be slightly more memory-efficient as it doesn't need to create tuples for each residual block.

2. **Migration Concerns**: If the model weights were saved using the original implementation, you might need to map the state dict keys when loading with the new implementation.

3. **Testing**: After implementing the fix, thoroughly test the model with a variety of audio inputs to ensure it works correctly.

## Next Steps After Implementation

1. Update the `get_audio_model` function to handle the new implementation if needed.

2. Ensure that any saved model weights are compatible with the new implementation.

3. Test the entire audio deepfake detection pipeline to verify that the fix resolves the issues.
