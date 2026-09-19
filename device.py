import subprocess


def probe_driver():
    """Read the NVIDIA driver version from the node.

    The driver, not the node's CUDA module, decides which wheel can load: JAX's pip
    CUDA wheels bundle their own runtime, and a mismatched wheel fails at import.

    Returns:
        The driver version string, e.g. ``"580.65.06"``.

    Raises:
        subprocess.CalledProcessError: If ``nvidia-smi`` is missing or fails.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.splitlines()[0].strip()


def cuda_major(driver, table):
    """Map a driver version to the CUDA major whose wheel it can load.

    Pure and total. The table is a parameter, never a literal inside the function, so
    a cluster with different drivers needs a different call site rather than a code
    change.

    Args:
        driver: Driver version string, as ``probe_driver`` returns it.
        table: Pairs of ``(minimum driver major, CUDA major)``, descending; the first
            match wins.

    Returns:
        The CUDA major version.

    Raises:
        ValueError: Carrying the driver string, if it cannot be parsed or if no entry
            matches.
    """
    head = driver.strip().split(".")[0]
    # isdigit() alone accepts non-ASCII numerals, but str.isascii() is 3.7+ and
    # job.sbatch may reach this under an older system python3, before any venv exists
    if not head or any(c not in "0123456789" for c in head):
        raise ValueError(f"unparseable NVIDIA driver version: {driver!r}")
    major = int(head)
    for min_driver_major, cuda in table:
        if major >= min_driver_major:
            return cuda
    raise ValueError(f"no CUDA major in table supports NVIDIA driver {driver!r}")


def require_accelerator(expect):
    """Fail loudly unless JAX is running on the expected backend.

    Turns a silent CPU fallback into a hard failure, which on a GPU allocation is the
    difference between a slow job and a wrong one.

    Args:
        expect: The required backend, e.g. ``"gpu"``.

    Raises:
        RuntimeError: Naming both the actual and the expected backend.
    """
    # jax imported late: job.sbatch runs this pre-venv
    import jax

    backend = jax.default_backend()
    if backend != expect:
        raise RuntimeError(f"JAX backend is {backend!r}, expected {expect!r}")
