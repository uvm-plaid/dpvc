# External Repository Changes Required

This document lists changes needed in external repositories (control-vc) for the ControlVC wrapper to work properly.

## Control-VC Repository

**Repository**: https://github.com/meiyingchen/ControlVC
**Local Path**: `~/repos/control-vc` (or wherever you cloned it)

### Required Change: Fix MPS Device Compatibility

**File**: `fairseq_feature_reader.py`
**Lines**: 22-26
**Reason**: MPS (Apple Silicon GPU) doesn't support all fairseq operators, causing HuBERT extraction to fail

**Original Code**:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
```

**Modified Code**:
```python
# Force CPU for compatibility - MPS doesn't support all fairseq operators
# DEVICE = torch.device("cuda" if torch.cuda.is_available()
#                       else "mps" if torch.backends.mps.is_available()
#                       else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### How to Apply

```bash
cd ~/repos/control-vc

# Edit the file
nano fairseq_feature_reader.py
# Or use your preferred editor

# Find line 22-24 and replace as shown above
```

### Verification

After applying the change, verify HuBERT extraction works:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from fairseq_feature_reader import HubertFeatureReader
reader = HubertFeatureReader(
    checkpoint_path='checkpoints/hubert_base_ls960.pt',
    layer=6
)
print('✓ HuBERT initialized successfully')
"
```

### Why Not Fork/PR?

This is a minimal compatibility fix specific to our use case. The upstream ControlVC repository may:
1. Have different device preferences for their workflows
2. Need MPS support for other parts of the codebase
3. Be managing this differently in unreleased versions

For our purposes, this local modification is sufficient. If you coordinate with multiple team members, consider:
- Creating a fork with this fix
- OR documenting this clearly in setup instructions (already done in CONTROLVC_SETUP.md)

### Tracking This Change

If you update the control-vc repository in the future:
1. Remember to reapply this fix
2. OR check if the upstream has addressed MPS compatibility
3. Always test HuBERT extraction after updating control-vc

### Alternative Solutions

If you prefer not to modify control-vc:

1. **Force CPU in environment**: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (slower)
2. **Use CUDA**: If you have an NVIDIA GPU, the original code works fine
3. **Fork control-vc**: Create a fork with this fix for your team

### Contact

If issues arise with this fix, contact: Steve
