# ARBM Deployment Troubleshooting Guide
**Date:** 2026-02-09
**Author:** Deepak Soni
**Environment:** OCI OKE with 4x NVIDIA A10 GPUs

---

## Issues Encountered and Resolutions

### 1. CoreDNS Pod Pending
**Problem:** CoreDNS pod stuck in Pending state
```
Warning: FailedScheduling - no nodes available to schedule pods
0/1 nodes are available: 1 node(s) had untolerated taint {nvidia.com/gpu: present}
```

**Root Cause:** GPU node has taint `nvidia.com/gpu=present:NoSchedule` but CoreDNS does not have toleration.

**Solution:**
```bash
kubectl patch deployment coredns -n kube-system --type=json \
  -p='[{"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}}]'
```

---

### 2. NVIDIA GPU Device Plugin CrashLoopBackOff
**Problem:** OKE default nvidia-gpu-device-plugin crashes with:
```
Incompatible strategy detected auto
If this is a GPU node, did you configure the NVIDIA Container Toolkit?
```

**Root Cause:** NVIDIA Container Toolkit not properly configured on node.

**Solution:** Enable GPU Operator (already installed but pending). Patch GPU Operator deployments:
```bash
kubectl patch deployment gpu-operator -n gpu-operator --type=json \
  -p='[{"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}}]'

kubectl patch deployment gpu-operator-node-feature-discovery-master -n gpu-operator --type=json \
  -p='[{"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}}]'
```

**Result:** GPU Operator deploys:
- nvidia-container-toolkit-daemonset
- nvidia-device-plugin-daemonset
- gpu-feature-discovery
- nvidia-dcgm-exporter

---

### 3. vLLM Image Pull Error
**Problem:** Container image fails with:
```
short name mode is enforcing, but image name vllm/vllm-openai:v0.6.6 returns ambiguous list
```

**Root Cause:** CRI-O requires full registry path.

**Solution:** Use full image path:
```yaml
image: docker.io/vllm/vllm-openai:v0.6.6
```

---

### 4. GLIBC Version Mismatch (Host vs Container)
**Problem:** When mounting host libs to container:
```
python3: /host-nvidia/libc.so.6: version GLIBC_2.35 not found
```

**Root Cause:** Oracle Linux 7.9 has GLIBC 2.17, but container needs 2.35+.

**Solution:** Do NOT mount host libs to LD_LIBRARY_PATH. Use GPU Operator which properly injects NVIDIA runtime.

---

### 5. GPU Resources Already Allocated
**Problem:**
```
0/1 nodes are available: 1 Insufficient nvidia.com/gpu
```

**Root Cause:** Old pods still holding GPU resources.

**Solution:**
```bash
# Find pods using GPUs
kubectl get pods -A -o wide --field-selector spec.nodeName=<NODE_IP>

# Delete old deployments/pods
kubectl delete deployment vllm-nemotron -n bench --force
kubectl delete pod <old-pod-name> -n <namespace> --force --grace-period=0
```

---

### 6. Model Architecture Not Supported
**Problem:**
```
ValueError: Model architectures ['NemotronHForCausalLM'] are not supported for now
```

**Root Cause:** vLLM 0.6.6 does not support NemotronH (Mamba-hybrid) architecture. Supported: NemotronForCausalLM (not H variant).

**Options:**
1. Use vLLM 0.7.0+ (requires newer image)
2. Use different supported model (Mixtral, Llama, etc.)
3. Use transformers library directly instead of vLLM
4. Use external API endpoints (OpenAI, Anthropic)

---

### 7. Mixtral OOM Error
**Problem:**
```
torch.OutOfMemoryError: CUDA out of memory
```

**Root Cause:** Mixtral-8x7B (46B params) too large for 4x A10 (96GB total).

**Solution:** Use smaller model or reduce max_model_len significantly.

---

## Working Configuration Summary

### GPU Node Status (After Fixes)
```
Node: 10.0.10.93
GPUs: 4x NVIDIA A10 (23028MB each)
Driver: 550.163.01
CUDA: 12.4
nvidia.com/gpu: 4 (allocatable)
```

### Required Tolerations for GPU Node
```yaml
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

### GPU Operator Pods (Running)
- gpu-operator
- gpu-feature-discovery
- nvidia-container-toolkit-daemonset
- nvidia-device-plugin-daemonset
- nvidia-dcgm-exporter
- nvidia-operator-validator

---

## Recommendations

1. **For NemotronH Model:** Need vLLM 0.7.0+ or use transformers directly
2. **Offline Environment:** Pre-pull all required images or use cached images
3. **Model Selection:** Consider smaller models that fit in 96GB (4x A10)
4. **Alternative:** Run benchmarks against external API endpoints

---

## Useful Commands

```bash
# Check GPU allocation
kubectl describe node <NODE> | grep -A8 'Allocated resources:'

# Check GPU operator status
kubectl get pods -n gpu-operator

# Check device plugin logs
kubectl logs -n gpu-operator nvidia-device-plugin-daemonset-<ID>

# Force delete stuck pods
kubectl delete pod <NAME> -n <NS> --force --grace-period=0
```
