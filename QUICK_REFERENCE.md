# Distributed Inference - Quick Reference

## Network Configuration
- **Studio 1 IP:** 10.201.20.10
- **Studio 2 IP:** 10.201.20.12
- **Network:** Thunderbolt (0.5ms latency)
- **Bandwidth:** ~25 MB/s

## Ports
- **50051** - Coordinator (Studio 1)
- **50052** - Workers (both studios)
- **8000** - API Server (Studio 1)

## Quick Commands

### Start Everything
```bash
./deploy_native_only.sh
```

### Check Status
```bash
./check_status.sh
./quick_validation.sh
```

### Test Network
```bash
./test_node_connectivity.sh
```

### Test Registration
```bash
python3 test_worker_registration.py
```

### View Logs
```bash
tail -f logs/coordinator.log
tail -f logs/api.log
tail -f logs/worker-studio1.log
ssh mac-studio-2 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'
```

### Stop Everything
```bash
pkill -f "python3 -m api.main"
pkill -f "python3 -m coordinator.main"
pkill -f "python3 -m worker.main"
ssh mac-studio-2 "pkill -f 'python3 -m worker.main'"
```

## Validation Results

✅ **Network:** Both nodes see each other  
✅ **Services:** Can deploy and run  
✅ **Registration:** Protocol works  
⚠️ **Workers:** Registration timing issues  
⚠️ **Distributed Layers:** Not yet implemented  

## Files Created

### Scripts
- `test_node_connectivity.sh` - Network test
- `quick_validation.sh` - Quick check
- `validate_nodes.sh` - Full validation
- `validate_distributed_layers.py` - Python validation
- `test_worker_registration.py` - Registration test
- `restart_workers.sh` - Restart workers

### Documentation
- `VALIDATION_RESULTS.md` - Detailed results
- `DISTRIBUTED_VALIDATION_README.md` - How-to guide
- `DISTRIBUTED_LAYER_INFERENCE_DESIGN.md` - Design doc
- `FINAL_VALIDATION_SUMMARY.md` - Summary
- `QUICK_REFERENCE.md` - This file

## Next Steps

1. Fix worker registration flow
2. Implement DistributedInferenceService
3. Create model partitioner
4. Implement activation transfer
5. Test distributed inference

See `DISTRIBUTED_LAYER_INFERENCE_DESIGN.md` for full implementation plan.
