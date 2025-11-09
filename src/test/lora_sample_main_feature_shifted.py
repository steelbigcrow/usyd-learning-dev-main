from __future__ import annotations

# Init startup path, change current path to startup python file folder
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

# import
# Enable lightweight monitoring (CSV + console) per Monitor Plan
# This single import auto-loads config and instruments server/client runtime
import usyd_learning.monitoring.auto_enable  # noqa: F401
from usyd_learning.ml_utils import console
from fl_lora_sample.lora_sample_entry_feature_shifted import SampleAppEntry


g_app = SampleAppEntry()


def main():
    # Load app config set from yaml file
    # Use MNIST ZP feature shift with homogeneous rank
    g_app.load_app_config(
        "./fl_lora_sample/convergence_experiment/special_lora_mnist/ZP/zp_mnist_feature_shift_round60_epoch1.yaml"
    )

    # Get training rounds (allow override via env for quick tests)
    training_rounds = g_app.training_rounds
    try:
        _env_rounds = os.environ.get("FL_TRAINING_ROUNDS", "")
        if _env_rounds:
            training_rounds = int(_env_rounds)
            console.info(f"[env] Override training rounds -> {training_rounds}")
    except Exception:
        pass

    # Select device dynamically: prefer CUDA, then MPS, else CPU
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    # Run app
    g_app.run(device, training_rounds)
    return


if __name__ == "__main__":
    # Initial console options
    console.set_log_level("all")  # Log level: error > warn > ok > info > out > all
    console.set_debug(True)  # True for display debug info

    # Set log path and name if needed
    console.set_console_logger(log_path="./log/", log_name="console_trace")
    console.set_exception_logger(log_path="./log/", log_name="exception_trace")
    console.set_debug_logger(log_path="./log/", log_name="debug_trace")

    console.enable_console_log(True)  # True for log console info to file by log level
    console.enable_exception_log(True)  # True for log exception info to file
    console.enable_debug_log(True)  # True for log debug info to file

    console.out("Simple FL program (feature-shifted)")
    console.out("======================= PROGRAM BEGIN ==========================")
    main()
    console.out("\n======================= PROGRAM END ============================")
    # console.wait_any_key()

