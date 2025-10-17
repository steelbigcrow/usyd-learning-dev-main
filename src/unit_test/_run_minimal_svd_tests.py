import os
import sys
import unittest

# Ensure project src is on sys.path for 'usyd_learning' imports
PROJECT_SRC = r"C:\Users\22270\Desktop\study\Sydney University\courses\Code Space\usyd-learning-dev-main\src"
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)


def main() -> int:
    suite = unittest.TestSuite()
    # Load specific minimal tests we just added
    modules = [
        "unit_test.unittest_aggregator_svd",
        "unit_test.unittest_lora_broadcast_utils",
    ]
    loader = unittest.defaultTestLoader
    for mod in modules:
        suite.addTests(loader.loadTestsFromName(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
