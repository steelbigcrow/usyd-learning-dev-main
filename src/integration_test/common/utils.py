import os
import sys
from typing import Tuple, Any


def ensure_startup(start_file: str) -> Tuple[str, str]:
    """
    Minimal startup bootstrap for tests:
    - chdir to the directory containing the test file
    - append 'src' and 'src/test' to sys.path so that 'usyd_learning' and
      'fl_lora_sample' modules can be imported

    Returns (src_dir, test_dir).
    """
    base_dir = os.path.dirname(os.path.abspath(start_file))
    # Align CWD to the test file directory (parity with startup_init.py behavior)
    os.chdir(base_dir)

    # Compute absolute paths
    src_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    test_dir = os.path.abspath(os.path.join(src_dir, "test"))

    if src_dir not in sys.path:
        sys.path.append(src_dir)
    if test_dir not in sys.path:
        sys.path.append(test_dir)

    return src_dir, test_dir


def run_scenario(config_path: str, rounds: int = 1, device: str = "cpu") -> Tuple[Any, Any, Any]:
    """
    Run a single federation scenario by loading a combo YAML and executing a
    few rounds. Returns (entry, runner, server_var).

    The caller should have invoked ensure_startup(__file__) so imports resolve.
    """
    # Lazy import after sys.path is prepared
    from fl_lora_sample.lora_sample_entry import SampleAppEntry

    app = SampleAppEntry()
    app.load_app_config(config_path)
    app.run(device, rounds)

    runner = app.fed_runner
    server_var = runner.server_node.node_var
    return app, runner, server_var


def run_scenario_skewed(config_path: str, rounds: int = 1, device: str = "cpu") -> Tuple[Any, Any, Any]:
    """
    Run a single scenario with the skewed long-tail entry that uses a decoupled
    partitioner for non-IID distribution. Returns (entry, runner, server_var).
    """
    from integration_test.common.skewed_entry import SkewedSampleAppEntry

    app = SkewedSampleAppEntry()
    app.load_app_config(config_path)
    app.run(device, rounds)

    runner = app.fed_runner
    server_var = runner.server_node.node_var
    return app, runner, server_var
