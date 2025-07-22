"""
Helpers for Evaluations
Copied and then Adapted from the KernelBench evaluation code
"""
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import torch
import torch.nn as nn
import subprocess
import random
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Dict, Union, Optional, Callable


def valid_vector(LL, bar=0.1, n_remove_top_bottom=1):
    """
    Check if the difference between max and min values in a list exceeds 10%
    After removing the top 2 and bottom 2 values from the list

    Args:
        LL (list): List of numeric values
        bar (float): Threshold for the difference ratio (default: 0.1)

    Returns:
        Tuple[bool, float]: (is_valid, diff_ratio)
               is_valid: False if max-min difference exceeds bar, True otherwise
               diff_ratio: The actual difference ratio
    """
    if not LL:
        return True, 0.0

    # Need at least 5 elements to remove top 2 and bottom 2 and still have values left
    if len(LL) < 5:
        print(f"not enough elements when computing valid_vector, 4 at least. no we have {len(LL)}")
        return True, 0.0

    # Sort the list to easily remove top 2 and bottom 2
    LL_sorted = sorted(LL)

    # Remove bottom 2 and top 2 values
    LL_trimmed = LL_sorted[n_remove_top_bottom:-n_remove_top_bottom]

    # Compute min and max from remaining values
    min_val = min(LL_trimmed)
    max_val = max(LL_trimmed)

    if min_val <= 0:
        return False, float('inf')

    diff_ratio = (max_val - min_val) / min_val
    return diff_ratio <= bar, diff_ratio


pst_tz = timezone(timedelta(hours=-8))

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def execute_model_with_timeout(
        model_src: str,
        context: Dict,
        timeout: float = 300.0,
        build_directory: Optional[str] = None,
        use_process_isolation: bool = False,
        info_string: str = ""
) -> Tuple[bool, str, Optional[float]]:
    """
    Execute model source code with a time limit.

    Args:
        model_src: Source code to execute (can be original_model_src or custom_model_src)
        context: Dictionary to execute the code in
        timeout: Maximum time in seconds to allow for execution (default: 300s = 5 minutes)
        build_directory: Optional build directory for CUDA extensions
        use_process_isolation: Use multiprocessing instead of threading (slower but more robust)

    Returns:
        Tuple[bool, str, Optional[float]]: (success, error_message, execution_time)
            - success: True if execution completed within timeout, False otherwise
            - error_message: Error details if execution failed, empty string if successful
            - execution_time: Time taken for execution in seconds, None if failed

    Note:
        ThreadPoolExecutor cannot interrupt blocking operations like time.sleep(),
        network requests, or infinite loops. The timeout detection works correctly,
        but background threads may continue running until the blocking operation completes.
        For CUDA code, this is usually not an issue as compilation errors are detected quickly.
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # Prepare source code with build directory if provided
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_src = (
                        "import os\n"
                        f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
                    ) + model_src

    # Static analysis for potentially problematic patterns
    potentially_hanging_patterns = [
        ('time.sleep(', 'time.sleep() calls'),
        ('requests.get(', 'network requests'),
        ('urllib.request.', 'URL requests'),
        ('input(', 'user input'),
        ('while True:', 'infinite loops'),
        ('subprocess.', 'subprocess calls'),
    ]

    detected_patterns = []
    for pattern, description in potentially_hanging_patterns:
        if pattern in model_src:
            detected_patterns.append(description)

    if detected_patterns:
        print(f"{info_prefix}[execute_model_with_timeout] WARNING: Detected potentially blocking operations:")
        for pattern in detected_patterns:
            print(f"{info_prefix}  - {pattern}")
        print(f"{info_prefix}[execute_model_with_timeout] These may cause hanging if they block indefinitely.")
        print(f"{info_prefix}[execute_model_with_timeout] Consider using use_process_isolation=True for risky code.")

        # Check for extremely problematic patterns that should be blocked
        blocking_patterns = ['time.sleep(', 'input(', 'while True:']
        should_block = any(pattern in model_src for pattern, _ in potentially_hanging_patterns
                           if pattern in blocking_patterns)

        if should_block and not use_process_isolation:
            error_msg = f"Code contains blocking patterns that may cause indefinite hanging: {detected_patterns}"
            print(f"{info_prefix}[execute_model_with_timeout] BLOCKING EXECUTION: {error_msg}")
            print(f"{info_prefix}[execute_model_with_timeout] Use use_process_isolation=True to override")
            return False, error_msg, None

    def _execute_code():
        """Helper function to execute the code in a separate thread"""
        try:
            compile(model_src, "<string>", "exec")
            exec(model_src, context)
            return True
        except Exception as e:
            raise e

    try:
        isolation_method = "process isolation" if use_process_isolation else "thread isolation"
        print(f"{info_prefix}Executing model code with {timeout}s timeout using {isolation_method}...")

        if use_process_isolation:
            # Use multiprocessing (more robust but has limitations with CUDA)
            import multiprocessing as mp
            print(
                f"{info_prefix}[execute_model_with_timeout] WARNING: Process isolation may not work well with CUDA contexts")

            def _execute_in_process():
                try:
                    compile(model_src, "<string>", "exec")
                    local_context = {}
                    exec(model_src, local_context)
                    return True
                except Exception as e:
                    raise e

            process = mp.Process(target=_execute_in_process)
            t1 = time.time()
            process.start()
            process.join(timeout=timeout)
            t2 = time.time()
            execution_time = t2 - t1

            if process.is_alive():
                print(f"{info_prefix}[execute_model_with_timeout] Process timeout - terminating")
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join()

                error_msg = f"Execution timeout after {execution_time:.6f} seconds"
                print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
                return False, error_msg, None

            if process.exitcode == 0:
                print(f"{info_prefix}Model code execution completed successfully")
                # Note: Process isolation doesn't update the context
                print(f"{info_prefix}[execute_model_with_timeout] Note: Context not updated due to process isolation")
                return True, "", execution_time
            else:
                error_msg = f"Process exited with code {process.exitcode}"
                return False, error_msg, None

        else:
            # Use threading (faster, works with CUDA, but can't interrupt blocking operations)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_code)
                try:
                    t1 = time.time()
                    future.result(timeout=timeout)
                    t2 = time.time()
                    execution_time = t2 - t1
                    print(f"{info_prefix}Model code execution completed successfully")
                    return True, "", execution_time

                except TimeoutError:
                    future.cancel()  # This won't stop blocking operations
                    elapsed_time = time.time() - t1
                    error_msg = f"Execution timeout after {elapsed_time:.6f} seconds"
                    print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
                    print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
                    print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
                    if detected_patterns:
                        print(
                            f"{info_prefix}[execute_model_with_timeout] Note: Background thread may still be running due to blocking operations")
                    return False, error_msg, None

    except SyntaxError as e:
        error_msg = f"Syntax Error: {e}"
        print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
        print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
        print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
        return False, error_msg, None

    except Exception as e:
        error_msg = f"Runtime Error: {e}"
        print(f"{info_prefix}[execute_model_with_timeout] {error_msg}")
        print(f"{info_prefix}[execute_model_with_timeout] Source code length: {len(model_src)} chars")
        print(f"{info_prefix}[execute_model_with_timeout] First 200 chars: {model_src[:200]}...")
        return False, error_msg, None


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def load_original_model_and_inputs(
        model_original_src: str, context: Dict, timeout: float = 300.0, info_string: str = ""
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement

    Args:
        model_original_src: Source code for the original model
        context: Dictionary to execute the code in
        timeout: Maximum time in seconds to allow for code execution (default: 300s = 5 minutes)
        info_string: Information string for consistent logging
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # Execute the model source code with timeout
    success, error_msg, execution_time = execute_model_with_timeout(
        model_src=model_original_src,
        context=context,
        timeout=timeout,
        build_directory=None,  # Original models typically don't need CUDA extensions
        info_string=info_string
    )

    if not success:
        print(f"{info_prefix}[load_original_model_and_inputs] Failed to execute original model code: {error_msg}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model(
        model_custom_src: str, context: Dict, build_directory: Optional[str] = None, timeout: float = 300.0,
        info_string: str = ""
) -> Optional[nn.Module]:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels

    Args:
        model_custom_src: Source code for the custom model
        context: Dictionary to execute the code in
        build_directory: Directory for CUDA extensions
        timeout: Maximum time in seconds to allow for code execution (default: 300s = 5 minutes)
        info_string: Information string for consistent logging
    """
    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # Execute the model source code with timeout
    success, error_msg, execution_time = execute_model_with_timeout(
        model_src=model_custom_src,
        context=context,
        timeout=timeout,
        build_directory=build_directory,
        info_string=info_string
    )

    if not success:
        print(f"{info_prefix}[load_custom_model] Failed to execute custom model code: {error_msg}")
        return None

    if execution_time is not None:
        print(f"{info_prefix}[load_custom_model] Model loaded successfully in {execution_time:.2f}s")

    ModelNew = context.get("ModelNew")

    # Debug: Show what's in the context
    print(f"{info_prefix}[load_custom_model] Context keys: {list(context.keys())}")
    print(f"{info_prefix}[load_custom_model] ModelNew from context: {ModelNew}")

    # Validate that ModelNew was properly defined
    if ModelNew is None:
        print(f"{info_prefix}[load_custom_model] Error: ModelNew was not defined in the custom model source code")
        print(
            f"{info_prefix}[load_custom_model] Make sure your custom model source includes: ModelNew = YourModelClass")
        print(
            f"{info_prefix}[load_custom_model] Available in context: {[k for k in context.keys() if not k.startswith('__')]}")
        return None

    if not callable(ModelNew):
        print(f"{info_prefix}Error: ModelNew is not callable (got {type(ModelNew)})")
        print(f"{info_prefix}Make sure ModelNew is a class that can be instantiated")
        return None

    # Additional validation - check if it's a class
    if not isinstance(ModelNew, type):
        print(f"{info_prefix}Error: ModelNew should be a class, got {type(ModelNew)}")
        print(f"{info_prefix}Example: class MyModel(nn.Module): ... then ModelNew = MyModel")
        return None

    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(curr_context: Dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_legacy(
        custom_model_src: str,
        verbose: bool = False,
        build_dir: Optional[os.PathLike] = None,
        timeout: float = 600.0,
) -> Tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str, str]: whether compilation is successful, stdout content as string, error message
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            # Use the robust timeout execution
            success, error_msg, execution_time = execute_model_with_timeout(
                model_src=custom_model_src,
                context=context,
                timeout=timeout,
                build_directory=build_dir
            )

            if not success:
                print(f"[Compilation] Failed to compile custom CUDA kernel: {error_msg}")
                return False, stdout_buffer.getvalue(), error_msg

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
            if execution_time is not None:
                print(f"[Compilation] Compilation took {execution_time:.2f}s")

    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
        custom_model_src: str,
        verbose: bool = False,
        build_dir: Optional[os.PathLike] = None,
        timeout: float = 600.0,
) -> Tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str, str]: whether compilation is successful, stdout content as string, error message
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            # Use the robust timeout execution
            success, error_msg, execution_time = execute_model_with_timeout(
                model_src=custom_model_src,
                context=context,
                timeout=timeout,
                build_directory=build_dir
            )

            if not success:
                print(f"[Compilation] Failed to compile custom CUDA kernel: {error_msg}")
                return False, stdout_buffer.getvalue(), error_msg

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
            if execution_time is not None:
                print(f"[Compilation] Compilation took {execution_time:.2f}s")

    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
        custom_model_src: str,
        verbose: bool = False,
        build_dir: Optional[os.PathLike] = None
) -> Tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
                               "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
                           ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(['python', tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode('utf-8'))
        print("[CPU Precompile] stderr: \n", stderr.decode('utf-8'))

    return returncode, stdout.decode('utf-8'), stderr.decode('utf-8')


def preliminary_speed_comparison(
        original_model_src: str,
        custom_model_src: str,
        timeout: float = 60.0,
        verbose: bool = False,
        device: Optional[torch.device] = None,
        info_string: str = ""
) -> Tuple[bool, Dict]:
    """
    Preliminary speed comparison to filter out obviously slow custom models.

    This function measures actual model inference time (forward pass with inputs),
    not compilation time. It runs original model 10 times, then tests if custom
    model can complete one forward pass within min(10*original_time, timeout).

    Args:
        original_model_src: Source code for the original/reference model
        custom_model_src: Source code for the custom model
        timeout: Maximum timeout in seconds (default: 60s)
        verbose: Whether to print detailed progress
        device: CUDA device to use (defaults to current device)

    Returns:
        tuple[bool, dict]: (passed, metadata)
            - passed: True if custom model inference completes within reasonable time
            - metadata: Dictionary with timing information and details
    """
    if device is None:
        raise Exception("Device is not set for preliminary_speed_comparison")

    # Define pst_tz at the beginning of the function
    pst_tz = timezone(timedelta(hours=-8))

    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    print(f"{info_prefix}[Preliminary Speed Filter] GPU DEVICE {device}")
    metadata = {
        "original_inference_times": [],
        "original_avg_inference_time": 0.0,
        "custom_inference_time": 0.0,
        "adaptive_timeout": 0.0,
        "final_timeout_used": 0.0,
        "original_trials": 10,
        "custom_trials": 1,
        "device": str(device)
    }

    # Step 1: Load original model

    context_original = {}
    success, error_msg, execution_time = execute_model_with_timeout(
        model_src=original_model_src,
        context=context_original,
        timeout=300.0  # Give enough time for loading
    )

    if not success:
        raise Exception(f"Original model loading failed: {error_msg}")

    # Get model components
    Model = context_original.get("Model")
    get_init_inputs = context_original.get("get_init_inputs")
    get_inputs = context_original.get("get_inputs")

    if not all([Model, get_init_inputs, get_inputs]):
        raise Exception("Original model missing required components (Model, get_init_inputs, get_inputs)")

    # Initialize original model
    init_inputs = get_init_inputs()
    if torch.cuda.is_available():
        init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    original_model = Model(*init_inputs)
    if torch.cuda.is_available():
        original_model = original_model.to(device)

    # Step 2: Measure original model inference time (10 trials)
    # if verbose:
    # print(f"{info_prefix}[Speed Filter] Measuring original model inference time (10 trials)...")
    # Get current Beijing time
    print(
        f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Original model inference start at gpu {device}")
    original_times = []
    with torch.no_grad():
        for trial in range(10):
            # Generate input for this trial
            inputs = get_inputs()
            if torch.cuda.is_available():
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

            # Time the forward pass
            start_time = time.time()
            _ = original_model(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            inference_time = time.time() - start_time

            original_times.append(inference_time)

    # Calculate average inference time for original model
    total_original_time = sum(original_times)
    print(
        f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Original model inference time: {total_original_time:.4f}s")
    try:
        # Step 3: Load custom model
        print(f"{info_prefix}[Preliminary Speed Filter] Loading custom model...")

        context_custom = {}
        success, error_msg, execution_time = execute_model_with_timeout(
            model_src=custom_model_src,
            context=context_custom,
            timeout=300.0  # Give enough time for loading/compilation
        )

        if not success:
            metadata["error"] = f"model loading failed: {error_msg}"
            return False, metadata

        ModelNew = context_custom.get("ModelNew")
        if ModelNew is None:
            metadata["error"] = "model missing ModelNew class"
            return False, metadata

        # Initialize custom model
        custom_model = ModelNew(*init_inputs)
        if torch.cuda.is_available():
            custom_model = custom_model.to(device)

        # Step 4: Calculate adaptive timeout for custom model inference
        # Use min(10 * original_inference_time, timeout)
        adaptive_timeout = min(10 * total_original_time, timeout)
        if total_original_time < 0.0001:
            adaptive_timeout = min(500 * total_original_time, timeout)
        metadata["adaptive_timeout"] = round(adaptive_timeout, 4)
        metadata["final_timeout_used"] = round(adaptive_timeout, 4)
        # print(f"total_original_time {10*total_original_time} adaptive_timeout {adaptive_timeout} timeout {timeout}")
        print(
            f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Adaptive timeout for custom inference: {adaptive_timeout:.4f}s")
        print(
            f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] (10 × {total_original_time:.4f}s = {10 * total_original_time:.4f}s, capped at {timeout}s)")

        # Step 5: Test custom model inference with timeout
        if verbose:
            print(f"{info_prefix}[Speed Filter] Testing custom model inference...")

        # Generate same type of input
        inputs = get_inputs()
        if torch.cuda.is_available():
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

        # Time the custom model inference with timeout
        start_time = time.time()

        def run_custom_inference():
            with torch.no_grad():
                result = custom_model(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device=device)
                return result

        # Use ThreadPoolExecutor for timeout control
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_custom_inference)
            try:
                result = future.result(timeout=adaptive_timeout)
                inference_time = time.time() - start_time
                metadata["custom_inference_time"] = round(inference_time, 4)
                metadata["passed"] = True

                return True, metadata

            except TimeoutError:
                inference_time = time.time() - start_time
                metadata["custom_inference_time"] = round(inference_time, 4)
                metadata["timeout_exceeded"] = True

                if verbose:
                    print(
                        f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ Custom model inference timeout after {inference_time:.4f}s")
                    print(
                        f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Exceeded adaptive timeout of {adaptive_timeout:.4f}s")

                metadata["error"] = f"Inference too slow or not even able to finish. 10x than the original model"
                return False, metadata

    except Exception as e:
        if verbose:
            print(
                f"{info_prefix}[Preliminary Speed Filter {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ Unexpected error: {e}")

        metadata["error"] = f"Unexpected error during speed comparison: {e}"
        return False, metadata


def check_kernel_correctness(
        original_model_src: str,
        custom_model_src: str,
        seed_num: int = 42,
        num_correct_trials: int = 5,
        verbose: bool = False,
        build_dir: Optional[os.PathLike] = None,
        device: Optional[torch.device] = None,
        timeout: float = 300.0,
        info_string: str = ""
) -> Tuple[bool, str, Dict]:
    """
    Check correctness of custom CUDA kernel against reference implementation.

    Args:
        original_model_src: Source code for the original/reference model
        custom_model_src: Source code for the custom CUDA kernel model
        seed_num: Base seed for reproducible testing
        num_correct_trials: Number of trials with different inputs to test
        verbose: Whether to print detailed progress
        build_dir: Directory for CUDA extensions
        device: CUDA device to run on (defaults to current device)
        timeout: Timeout for model loading in seconds

    Returns:
        tuple[bool, str, dict]: (success, error_message, metadata)
            - success: True if all correctness trials pass
            - error_message: Error details if failed, empty string if successful
            - metadata: Dictionary with trial details and statistics
    """
    if device is None:
        raise Exception("Device is not set for check_kernel_correctness")

    if not torch.cuda.is_available():
        return False, "CUDA is not available", {}

    # Define pst_tz at the beginning of the function
    pst_tz = timezone(timedelta(hours=-8))

    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # Set CUDA device
    torch.cuda.set_device(device)

    metadata = {
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device=device),
        "num_trials": num_correct_trials,
        "trials_passed": 0,
        "trials_failed": 0,
        "max_difference": [],
        "avg_difference": []
    }

    if verbose:
        print(f"{info_prefix}[Correctness] Starting correctness check on device: {device}")
        print(f"{info_prefix}[Correctness] Running {num_correct_trials} trials")

    # Load original model
    context_original = {}
    if verbose:
        print(f"{info_prefix}[Correctness] Loading original model...")

    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            original_model_src, context_original, timeout=timeout, info_string=info_string
        )
        if Model is None:
            return False, "Failed to load original model", metadata

        # Initialize original model
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        with torch.no_grad():
            set_seed(seed_num)
            original_model = Model(*init_inputs).to(device)

    except Exception as e:
        return False, f"Failed to initialize original model: {e}", metadata

    # Load custom model
    context_custom = {}
    if verbose:
        print(f"{info_prefix}[Correctness] Loading custom model...")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable device-side assertions
        ModelNew = load_custom_model(custom_model_src, context_custom, build_dir, timeout=timeout,
                                     info_string=info_string)
        if ModelNew is None:
            return False, "Failed to load custom model", metadata

        # Initialize custom model
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device)

        torch.cuda.synchronize(device=device)

    except Exception as e:
        return False, f"Failed to initialize custom model: {e}", metadata

    # Run correctness trials
    if verbose:
        print(f"{info_prefix}[Correctness] Running {num_correct_trials} correctness trials...")

    # Generate trial seeds deterministically
    torch.manual_seed(seed_num)
    trial_seeds = [torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(num_correct_trials)]

    pass_count = 0

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]

            # if verbose:
            #     print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial + 1}/{num_correct_trials} (seed: {trial_seed})")

            try:
                # Generate inputs for this trial
                set_seed(trial_seed)
                inputs = get_inputs()
                inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

                # Run original model
                set_seed(trial_seed)
                original_model.eval()
                original_output = original_model(*inputs)
                torch.cuda.synchronize(device=device)

                # Run custom model
                set_seed(trial_seed)
                custom_model.eval()
                custom_output = custom_model(*inputs)
                torch.cuda.synchronize(device=device)

                # Check output shapes
                if original_output.shape != custom_output.shape:
                    error_msg = f"Shape mismatch to the original model"
                    metadata["trials_failed"] += 1
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ❌ {error_msg}")
                    return False, error_msg, metadata

                # Check output values
                if not torch.allclose(original_output, custom_output, atol=1e-02, rtol=1e-02):
                    max_diff = torch.max(torch.abs(original_output - custom_output)).item()
                    avg_diff = torch.mean(torch.abs(original_output - custom_output)).item()

                    metadata["max_difference"].append(f"{max_diff:.6f}")
                    metadata["avg_difference"].append(f"{avg_diff:.6f}")
                    metadata["trials_failed"] += 1

                    error_msg = f"Value mismatch to the original model"
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ❌ {error_msg}")
                    return False, error_msg, metadata
                else:
                    # Trial passed
                    pass_count += 1
                    metadata["trials_passed"] += 1
                    # if verbose:
                    #     print(f"{info_prefix}[Correctness] ✅ Trial {trial + 1} passed")

            except Exception as e:
                metadata["trials_failed"] += 1
                error_msg = f"Runtime error in trial {trial + 1}: {e}"
                if verbose:
                    print(
                        f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ {error_msg}")
                return False, error_msg, metadata

    # Final validation
    if pass_count == num_correct_trials:
        if verbose:
            print(
                f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ✅ All {pass_count}/{num_correct_trials} trials passed!")

        # Cleanup
        graceful_eval_cleanup(context_original, device)
        graceful_eval_cleanup(context_custom, device)

        return True, "", metadata
    else:
        error_msg = f"Only {pass_count}/{num_correct_trials} trials passed"
        if verbose:
            print(f"{info_prefix}[Correctness {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ {error_msg}")
        return False, error_msg, metadata


def eval_kernel_against_ref(
        original_model_src: str,
        custom_model_src: str,
        seed_num: int = 42,
        num_perf_trials: int = 10,
        verbose: bool = False,
        build_dir: Optional[os.PathLike] = None,
        device: Optional[torch.device] = None,  # have to run on GPU
        info_string: str = ""
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
            (score, total_elapsed_time_seconds, avg_original_time_seconds, avg_custom_time_seconds, message)
            - score: original_model_time / custom_model_time (higher is better, >1.0 means speedup), or None if failed
            - total_elapsed_time_seconds: Total GPU execution time in seconds
            - avg_original_time_seconds: Average original model time per trial in seconds
            - avg_custom_time_seconds: Average custom model time per trial in seconds
            - message: Error message if failed, "Success" if successful
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # Define pst_tz at the beginning of the function
    pst_tz = timezone(timedelta(hours=-8))

    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"{info_prefix}[Eval] Start Evalulation! on device: {device}")
        print(f"{info_prefix}[Eval] Loading Original Model")

    result = load_original_model_and_inputs(
        original_model_src, context, info_string=info_string
    )
    if result is None:
        return None, None, None, None, "Failed to load original model"

    Model, get_init_inputs, get_inputs = result
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print(f"{info_prefix}[Eval] Original Model Loaded")
    if verbose:
        print(f"{info_prefix}[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir, info_string=info_string)

        # Debug: Check what load_custom_model returned
        if verbose:
            print(f"{info_prefix}[DEBUG] load_custom_model returned: {ModelNew} (type: {type(ModelNew)})")

        # Validate ModelNew before proceeding
        if ModelNew is None:
            print(f"{info_prefix}ERROR: load_custom_model returned None - check the model source code")
            print(f"{info_prefix}The custom model source must define: ModelNew = YourModelClass")
            return None, None, None, None, "ModelNew is None"

        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"{info_prefix}Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        return None, None, None, None, "Failed to compile custom CUDA kernel"

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print(f"{info_prefix}[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"{info_prefix}Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        return None, None, None, None, "Failed to load custom CUDA kernel with New Model"

    # Handle case where num_correct_trials is 0 (skip correctness check)

    if verbose:
        print(f"{info_prefix}[Eval] Measuring Performance")

    # Move models to the correct device for performance measurement
    original_model = original_model.to(device)
    custom_model = custom_model.to(device)

    original_times = []
    custom_times = []

    # Warmup
    for _ in range(3):
        inputs = get_inputs()
        inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        original_model(*inputs)
        custom_model(*inputs)
        torch.cuda.synchronize(device=device)

    if verbose:
        print(
            f"{info_prefix}[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, trials: {num_perf_trials}")

    t1 = time.time()
    with torch.no_grad():
        for trial in range(num_perf_trials):
            # Generate one random input for this trial - SAME input will be used for both models
            inputs = get_inputs()
            inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            # Randomize execution order to eliminate systematic bias
            run_original_first = random.choice([True, False])

            # IMPORTANT: Get the current CUDA stream to ensure events and execution are on the same stream
            current_stream = torch.cuda.current_stream(device=device)

            if run_original_first:
                # Time original model first
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                with torch.cuda.stream(current_stream):
                    start_event.record(current_stream)
                    original_model(*inputs)
                    torch.cuda.synchronize(device=device)
                    end_event.record(current_stream)
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)

                # Time custom model second
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                with torch.cuda.stream(current_stream):
                    start_event.record(current_stream)
                    custom_model(*inputs)
                    torch.cuda.synchronize(device=device)
                    end_event.record(current_stream)
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)
            else:
                # Time custom model first
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                with torch.cuda.stream(current_stream):
                    start_event.record(current_stream)
                    custom_model(*inputs)
                    torch.cuda.synchronize(device=device)
                    end_event.record(current_stream)
                torch.cuda.synchronize(device=device)
                custom_time = start_event.elapsed_time(end_event)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                with torch.cuda.stream(current_stream):
                    start_event.record(current_stream)
                    original_model(*inputs)
                    torch.cuda.synchronize(device=device)
                    end_event.record(current_stream)
                torch.cuda.synchronize(device=device)
                original_time = start_event.elapsed_time(end_event)

            original_times.append(original_time)
            custom_times.append(custom_time)
    t2 = time.time()

    # Calculate averages and score
    # CUDA events return time in milliseconds, convert to seconds for consistency
    if len(original_times) == 0 or len(custom_times) == 0:
        return None, None, None, None, "No timing data collected"

    avg_original_time_ms = sum(original_times) / len(original_times)
    avg_custom_time_ms = sum(custom_times) / len(custom_times)
    avg_original_time_seconds = avg_original_time_ms / 1000.0
    avg_custom_time_seconds = avg_custom_time_ms / 1000.0

    # Protect against division by zero
    if avg_custom_time_ms == 0:
        return None, None, None, None, "Custom model execution time is zero"

    score = avg_original_time_ms / avg_custom_time_ms  # Use milliseconds for score calculation
    total_elapsed_time = (sum(original_times) + sum(custom_times)) / 1000.0  # Convert from milliseconds to seconds

    if verbose:
        print(
            f"{info_prefix}[Results {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Original avg: {avg_original_time_ms:.3f}ms")
        print(
            f"{info_prefix}[Results {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Custom avg: {avg_custom_time_ms:.3f}ms")
        print(
            f"{info_prefix}[Results {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Score (original/custom): {score:.3f}")
        if score > 1.0:
            speedup = score
            print(
                f"{info_prefix}[Results {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Speedup: {speedup:.2f}x faster")
        elif score < 1.0:
            slowdown = 1.0 / score
            print(
                f"{info_prefix}[Results {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Slowdown: {slowdown:.2f}x slower")
        else:
            print(f"{info_prefix}[Results] Same performance")

    graceful_eval_cleanup(context, device)
    return score, total_elapsed_time, avg_original_time_seconds, avg_custom_time_seconds, "Success"


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
        level_name: str, problem_id: int, dataset: List[str], baseline_time_filepath: str
) -> Dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: List[float], device: Optional[torch.device] = None) -> Dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


def get_available_gpus():
    """Get list of available GPU device IDs"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def round_toward_one(value: float) -> float:
    """
    Round to 2 decimal places with bias toward 1.00.

    Examples:
        1.013 → 1.01
        1.028 → 1.02 (bias down toward 1.00)
        0.993 → 1.00 (bias up toward 1.00)
        0.991 → 1.00 (bias up toward 1.00)

    Args:
        value (float): Input value to round

    Returns:
        float: Rounded value biased toward 1.00
    """
    # Convert to 2 decimal precision
    rounded = round(value, 2)

    # If very close to 1.00 (within 0.01), round to 1.00
    if abs(rounded - 1.00) <= 0.01:
        return 1.00

    # For values > 1.00, apply slight downward bias
    if value > 1.00:
        # Get the third decimal place
        third_decimal = int((value * 1000) % 10)

        # If third decimal is 5 or higher, normally would round up
        # But we bias down toward 1.00 for values close to 1.00
        if 1.00 < value < 1.05 and third_decimal >= 5:
            # Round down instead of up (bias toward 1.00)
            return round(value - 0.005, 2)

    # For values < 1.00, apply slight upward bias toward 1.00
    elif value < 1.00:
        # If very close to 1.00, round up
        if value >= 0.99:
            return 1.00

    # Default rounding for other cases
    return rounded


def eval_pipeline(
        original_model_src: str,
        custom_model_src: str,
        num_correct_trials: int,
        num_perf_trials: int,
        global_n_trials: int,
        gpu_index: int,
        verbose: bool = False,
        log_path: str = None,
        max_time: float = None,
        use_process_isolation: bool = False,
        info_string="",
        valid_bar=0.15
):
    pst_tz = timezone(timedelta(hours=-8))

    # Format info_string for consistent display
    info_prefix = f"[{info_string}] " if info_string else ""

    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] median_comparison_pipeline start")
    if log_path is not None:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Writing log to {log_path}")
    current_time = datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, "w") as write_log:
        print(f"in log_path open and write {log_path}")
        write_log.write(
            json.dumps({"info_string": info_string, "start_time": current_time, "code": custom_model_src}) + "\n")
        # write_log.write(json.dumps({"info_string": info_string, "start_time": current_time, "custom_model_src": custom_model_src}) + "\n")
        write_log.flush()

    # step 1: check whether the model can be executed and compiled
    print(
        f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 1: check whether the model can be executed and compiled")
    context = {}
    success_original, error_msg, execution_time = execute_model_with_timeout(
        model_src=original_model_src,
        context=context,
        timeout=30.0,  # 30 seconds should be enough
        use_process_isolation=use_process_isolation,
        info_string=info_string
    )
    if not success_original:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": f"Original model compilation failed: {error_msg}",
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, f"Original model compilation failed: {error_msg}"

    success_custom, error_msg, execution_time = execute_model_with_timeout(
        model_src=custom_model_src,
        context={},  # Use fresh context for custom model
        timeout=100,  # Give enough time for CUDA compilation with minimum 30s
        use_process_isolation=use_process_isolation,
        info_string=info_string
    )
    if not success_custom:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": "fail to compile or execute",
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, "Custom model compilation failed"
    else:
        log_dict_ = {
            "info_string": info_string,
            "info": "stage1:Compile Success",
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "error": False,
            "done": False
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
    # step 2: preliminary speed check
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 2: preliminary speed check")
    device = torch.device(f'cuda:{gpu_index}')
    time1 = time.time()
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 3: correctness check")
    time1 = time.time()
    correctness_passed, error_msg, correctness_metadata = check_kernel_correctness(
        original_model_src=original_model_src,
        custom_model_src=custom_model_src,
        num_correct_trials=num_correct_trials,
        verbose=verbose,
        device=device,
        info_string=info_string
    )
    time2 = time.time()
    if not correctness_passed:
        log_dict_ = {
            "info_string": info_string,
            "error_msg": error_msg,
            "error": True,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        return None, error_msg
    else:
        log_dict_ = {
            "info_string": info_string,
            "info": "stage3:Correctness Check Success",
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "error": False,
            "done": False,
            "duration": time2 - time1,
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()

    log_dict_ = {
        "info_string": info_string,
        "info": "stage4:Performance Evaluation",
        "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
        "error": False,
        "done": False
    }
    with open(log_path, "a") as write_log:
        write_log.write(json.dumps(log_dict_) + "\n")
        write_log.flush()
    scores = []
    list_gpu_execution_time = []
    # Run global_n_trials sequential evaluations
    start_time = time.time()
    print(f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 4: performance evaluation")
    for trial in range(global_n_trials):
        print(
            f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] step 4: performance evaluation, trial {trial + 1}/{global_n_trials}")
        # Run single evaluation
        time1 = time.time()
        score, gpu_execution_time, avg_original_time, avg_custom_time, error_msg = eval_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            seed_num=42 + trial,  # Different seed for each trial
            num_perf_trials=num_perf_trials,
            verbose=False,  # Keep individual trials quiet unless overall verbose
            build_dir=None,
            device=device,
            info_string=info_string
        )
        list_gpu_execution_time.append(gpu_execution_time)
        if score is None:
            error_msg = f"fail to inference"
            log_dict_ = {
                "info_string": info_string,
                "trial": trial,
                "gpu_index": gpu_index,
                "score": score,
                "error_msg": error_msg,
                "error": True,
                "done": True
            }
            with open(log_path, "a") as write_log:
                write_log.write(json.dumps(log_dict_) + "\n")
                write_log.flush()
            return None, error_msg
        time2 = time.time()
        log_dict_ = {
            "info_string": info_string,
            "n_trial": num_perf_trials,
            "trial": trial,
            "gpu_index": gpu_index,
            "score": score,
            "time": datetime.now(timezone(timedelta(hours=-8))).strftime('%Y-%m-%d %H:%M:%S'),
            "gpu_execution_time": gpu_execution_time,
            "ave_gpu_execution_time": gpu_execution_time / num_perf_trials,
            "avg_original_time": avg_original_time,
            "avg_custom_time": avg_custom_time,
            "done": False,
            "duration": time2 - time1,
            "error": False
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_) + "\n")
            write_log.flush()
        scores.append(score)

        print(
            f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial + 1}: {score:.4f} at gpu {gpu_index}")

    if len(scores) == 0:
        print(
            f"{info_prefix}[Score {datetime.now(pst_tz).strftime('%Y-%m-%d %H:%M:%S')}] ❌ No trials completed successfully")
        log_dict_empty = {
            "info_string": info_string,
            "error": True,
            "error_msg": "No trials completed successfully",
            "completed_trials": 0,
            "done": True
        }
        with open(log_path, "a") as write_log:
            write_log.write(json.dumps(log_dict_empty) + "\n")
            write_log.flush()
        return None, "No trials completed successfully"

    # Calculate median score and apply custom rounding
    # raw_median = float(np.median(scores))
    raw_mean = float(np.mean(scores))
    mean_score = round(raw_mean, 3)

    raw_median = float(np.median(scores))
    median_score = round_toward_one(raw_median)

    std = float(np.std(scores))

    # Round all scores in the list to 4 decimal places for consistency
    rounded_scores = [round(score, 4) for score in scores]

    # Record final elapsed time
    total_elapsed_time = time.time() - start_time
    if_valid, diff_ratio = valid_vector(scores, valid_bar)
    n_all_trials = num_perf_trials * global_n_trials
    log_dict_ = {
        "info_string": info_string,
        "median_score": median_score,
        "mean_score": mean_score,
        "rounded_scores": rounded_scores,
        "raw_median": raw_median,
        "raw_mean": raw_mean,
        "scores_sorted": sorted(scores),
        "completed_trials": len(scores),
        "total_trials": global_n_trials,
        "n_all_trials_trials": n_all_trials,
        "total_elapsed_time": total_elapsed_time,
        "total_gpu_execution_time": sum(list_gpu_execution_time),
        "ave_gpu_execution_time": sum(list_gpu_execution_time) / n_all_trials,
        "error": False,
        "done": True,
        "scores": [round(score, 4) for score in scores],
        "std": std,
        "if_valid": if_valid,
        "diff_ratio": diff_ratio
    }
    with open(log_path, "a") as write_log:
        write_log.write(json.dumps(log_dict_) + "\n")
        write_log.flush()

    if verbose:
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        trials_completed = len(scores)
        print(f"\n{info_prefix}[Score] 📊 Results from {trials_completed}/{global_n_trials} trials:")
        print(f"{info_prefix}  - Total time: {total_elapsed_time:.2f}s")
        if max_time is not None and total_elapsed_time >= max_time:
            print(f"{info_prefix}  - Status: TIMEOUT (reached {max_time}s limit)")
        print(f"{info_prefix}  - Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"{info_prefix}  - Raw Median: {raw_median:.4f}")
        print(f"{info_prefix}  - Final Median: {median_score:.2f}")
        print(f"{info_prefix}  - Mean:   {mean_score:.4f}")
        print(f"{info_prefix}  - Std:    {std_score:.4f}")
        print(f"{info_prefix}  - Range:  [{min_score:.4f}, {max_score:.4f}]")

        # Stability assessment
        cv = (std_score / mean_score) * 100 if mean_score > 0 else 0
        print(
            f"{info_prefix}  - CV:     {cv:.2f}% {'(stable)' if cv < 1.0 else '(variable)' if cv < 5.0 else '(unstable)'}")

    return median_score, rounded_scores


def load_cuda_file(PATH_TO_CUDA_FILE):
    if not os.path.exists(PATH_TO_CUDA_FILE):
        raise Exception(f"{PATH_TO_CUDA_FILE} not found")
    with open(PATH_TO_CUDA_FILE, "r") as f:
        ref_cuda_file = json.load(f)
    cuda_dict_ = {}
    for level, level_items in ref_cuda_file.items():
        cuda_dict_[int(level)] = {}
        for item in level_items:
            task_id = int(item["task_id"])
            ref_code = item["ref_code"]
            custom_code = item["custom_code"]
            cuda_dict_[int(level)][task_id] = (ref_code, custom_code)
    return cuda_dict_


if __name__ == "__main__":
    PATH_TO_CUDA_FILE = "optimized_cuda_code/rtx_3090.json"
    cuda_data_folder = os.path.dirname(PATH_TO_CUDA_FILE)

    cuda_dict_ = load_cuda_file(PATH_TO_CUDA_FILE)
    level_id = 3
    task_id = 42
    ref_code, custom_code = cuda_dict_[level_id][task_id]
    print(custom_code)
    output_path = os.path.join(cuda_data_folder, f"{level_id}_{task_id}_eval.json")
    print(f"eval results output to {output_path}")
    eval_pipeline(
        original_model_src=ref_code,
        custom_model_src=custom_code,
        num_correct_trials=100,
        num_perf_trials=100,
        global_n_trials=7,
        gpu_index=0,
        verbose=False,
        log_path=output_path,
        max_time=1800
    )
    print(f"log_path: {output_path}")
    print(f"log_path: {output_path}")
