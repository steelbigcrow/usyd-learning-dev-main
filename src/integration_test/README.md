Integration Tests for AdaLoRA + Aggregators

This folder hosts end-to-end integration tests that exercise:
- AdaLoRA trainers on clients
- Server-side aggregation with RBLA, Zero-Pad (ZP), and SVD
- Bridging between PEFT AdaLoRA state_dicts and plain-LoRA keys during aggregation

Layout
- common/utils.py: helpers to bootstrap imports and run a scenario
- adalora_rbla/: YAML combo + unittest for RBLA aggregation
- adalora_zp/: YAML combo + unittest for Zero-Pad aggregation (uses RBLA strategies for bridging)
- adalora_svd/: YAML combo + unittest for SVD aggregation and strategies

Run
- Single test (Windows venv):
  C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe -m unittest -v src/integration_test/adalora_rbla/test_adalora_rbla_integration.py
- All tests:
  C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\Capstone\.venv\Scripts\python.exe -m unittest discover -s src/integration_test -p "test_*.py" -v

Notes
- Uses MNIST with download enabled; prepare .dataset or adjust YAMLs to speed up.
- Keep rounds=1 in tests to minimize runtime while still validating the pipeline.

