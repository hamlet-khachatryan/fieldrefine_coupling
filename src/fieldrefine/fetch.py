"""Stage 1: fetch one PDB entry and cache coordinates, structure factors and MTZ.

All network and disk I/O in the pipeline that is not a ``read_*`` / ``write_*`` function lives here.
"""

import shutil
import urllib.error
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

RCSB_DOWNLOAD = "https://files.rcsb.org/download/"
XRAY_METHOD = "X-RAY DIFFRACTION"
FREE_FLAG_LABELS = (
    "FreeR_flag", "FREE", "Free", "FreeRflag", "RFREE", "R-free-flags", "FLAG", "TEST",
)
TIMEOUT_S = 120


class FetchError(RuntimeError):
    pass


def normalise_id(pdb_id):
    """Normalise a PDB identifier to RCSB's uppercase form.

    Args:
        pdb_id: Four-character PDB entry code, in any case.

    Returns:
        The identifier uppercased, e.g. ``"6LU7"``.

    Raises:
        ValueError: If it is not four alphanumeric characters starting with a digit.
    """
    pid = str(pdb_id).strip().upper()
    if len(pid) != 4 or not pid.isalnum() or not pid[0].isdigit():
        raise ValueError(f"not a PDB ID: {pdb_id!r}")
    return pid


def cache_paths(pdb_id, root="data"):
    """Give the cache locations for one entry without touching the disk.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.

    Returns:
        Dict with the normalised ``id``, the entry ``dir``, and the ``cif``,
        ``sf_cif`` and ``mtz`` paths.
    """
    pid = normalise_id(pdb_id)
    directory = Path(root) / pid.lower()
    return {
        "id": pid,
        "dir": directory,
        "cif": directory / f"{pid}.cif",
        "sf_cif": directory / f"{pid}-sf.cif",
        "mtz": directory / f"{pid}.mtz",
    }


def is_cached(path):
    """Report whether a cached file is usable.

    Args:
        path: File to test.

    Returns:
        True if the file exists and is non-empty. A zero-byte file counts as
        absent, so an interrupted write is re-fetched rather than trusted.
    """
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def download(url, dest, timeout=TIMEOUT_S):
    """Download a URL to a path, skipping the transfer if it is already cached.

    Writes to ``<dest>.part`` and renames on completion, so an interrupted
    download never leaves a file that looks cached.

    Args:
        url: Source URL.
        dest: Destination path.
        timeout: Socket timeout in seconds.

    Returns:
        The destination path.

    Raises:
        FetchError: On an HTTP error, an unreachable host, or an empty body.
    """
    dest = Path(dest)
    if is_cached(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{url} unreachable: {exc.reason}") from exc
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{url} returned an empty file")
    partial.replace(dest)
    return dest


def read_experimental_methods(cif_path):
    """Read the deposited experimental methods from a coordinate mmCIF.

    Args:
        cif_path: Path to the coordinate mmCIF.

    Returns:
        The ``_exptl.method`` values, e.g. ``["X-RAY DIFFRACTION"]``.
    """
    block = gemmi.cif.read(str(cif_path)).sole_block()
    return [gemmi.cif.as_string(v) for v in block.find_values("_exptl.method")]


def read_refln_block(sf_cif_path):
    """Select the merged reflection block from a structure-factor mmCIF.

    Args:
        sf_cif_path: Path to the structure-factor mmCIF.

    Returns:
        The first merged ``gemmi.ReflnBlock`` in the file.

    Raises:
        FetchError: If the file holds no reflection block, or only unmerged data.
    """
    blocks = list(gemmi.as_refln_blocks(gemmi.cif.read(str(sf_cif_path))))
    if not blocks:
        raise FetchError(f"{sf_cif_path} contains no reflection block")
    merged = [b for b in blocks if b.is_merged()]
    if not merged:
        raise FetchError(f"{sf_cif_path} contains only unmerged data; the pipeline needs merged F")
    return merged[0]


def read_mtz(mtz_path):
    """Read a cached MTZ.

    Every downstream stage reads reflections through this function, so a change
    in gemmi's reader is confined to this module.

    Args:
        mtz_path: Path to the MTZ.

    Returns:
        The ``gemmi.Mtz`` object.
    """
    return gemmi.read_mtz_file(str(mtz_path))


def write_mtz(sf_cif_path, mtz_path):
    """Convert a structure-factor mmCIF to MTZ, unless it is already cached.

    Args:
        sf_cif_path: Path to the structure-factor mmCIF.
        mtz_path: Destination MTZ path.

    Returns:
        The MTZ path.
    """
    mtz_path = Path(mtz_path)
    if is_cached(mtz_path):
        return mtz_path
    mtz = gemmi.CifToMtz().convert_block_to_mtz(read_refln_block(sf_cif_path))
    mtz_path.parent.mkdir(parents=True, exist_ok=True)
    partial = mtz_path.with_name(mtz_path.name + ".part")
    mtz.write_to_file(str(partial))
    partial.replace(mtz_path)
    return mtz_path


def free_flag_label(mtz):
    """Find the free-flag column, preferring the conventional labels.

    Args:
        mtz: Reflection file to inspect.

    Returns:
        The column label carrying the work/free split, e.g. ``"FreeR_flag"``.

    Raises:
        FetchError: If no free-flag column is present. Flags are never
            generated: a fresh test set has already been seen by the deposited
            model, which would make delta_R_free meaningless.
    """
    labels = mtz.column_labels()
    for name in FREE_FLAG_LABELS:
        if name in labels:
            return name
    try:
        column = mtz.rfree_column()
    except Exception:
        column = None
    label = getattr(column, "label", None)
    if label:
        return label
    raise FetchError(
        f"no free-flag column among {labels}; the pipeline requires deposited free flags"
    )


def free_flag_summary(mtz, label):
    """Count the reflections carrying each distinct free-flag value.

    Args:
        mtz: Reflection file to inspect.
        label: Free-flag column label.

    Returns:
        Mapping of flag value to reflection count, e.g. ``{0: 998, 1: 962}``.

    Raises:
        FetchError: If any flag is missing or non-integral. The convention
            itself is resolved later, by ``prepare.free_flag_convention``.
    """
    values = np.asarray(mtz.column_with_label(label).array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        raise FetchError(
            f"free-flag column {label!r} has {values.size - finite.size} missing values; "
            "the work/free split must be defined for every measured reflection"
        )
    if not np.allclose(finite, np.round(finite)):
        raise FetchError(f"free-flag column {label!r} is not integral")
    unique, counts = np.unique(np.round(finite).astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(unique, counts)}


def amplitude_labels(mtz):
    """Find the amplitude and sigma columns.

    Prefers the conventional pairs, then falls back to the first column of MTZ
    type ``F`` and type ``Q``.

    Args:
        mtz: Reflection file to inspect.

    Returns:
        Tuple of the amplitude and sigma labels, e.g. ``("FP", "SIGFP")``.

    Raises:
        FetchError: If the deposition carries intensities only.
    """
    labels = mtz.column_labels()
    for f_label, sig_label in (("FP", "SIGFP"), ("F", "SIGF"), ("F-obs", "SIGF-obs")):
        if f_label in labels and sig_label in labels:
            return f_label, sig_label
    amplitudes = [c.label for c in mtz.columns_with_type("F")]
    sigmas = [c.label for c in mtz.columns_with_type("Q")]
    if not amplitudes or not sigmas:
        raise FetchError(
            f"no amplitude/sigma pair among {labels}; "
            "intensity-only depositions are not supported"
        )
    return amplitudes[0], sigmas[0]


def fetch(pdb_id, root="data"):
    """Download, convert and validate one PDB entry.

    Idempotent: a second call against a populated cache performs no network
    access. Validates that the entry is X-ray, has structure factors, and has a
    free-flag column with more than one distinct value.

    Args:
        pdb_id: PDB entry code.
        root: Cache root directory.

    Returns:
        The ``cache_paths`` mapping extended with ``methods``, ``free_label``,
        ``free_flag_counts``, ``f_label``, ``sig_label``, ``spacegroup``,
        ``cell``, ``d_min``, ``d_max`` and ``n_reflections``.

    Raises:
        FetchError: If the entry is not X-ray, has no deposited structure
            factors, or has no usable test set.
    """
    paths = cache_paths(pdb_id, root)
    pid = paths["id"]
    download(f"{RCSB_DOWNLOAD}{pid}.cif", paths["cif"])
    try:
        download(f"{RCSB_DOWNLOAD}{pid}-sf.cif", paths["sf_cif"])
    except FetchError as exc:
        raise FetchError(f"{pid} has no deposited structure factors ({exc})") from exc

    methods = read_experimental_methods(paths["cif"])
    if XRAY_METHOD not in methods:
        raise FetchError(f"{pid} is not X-ray: _exptl.method = {methods}")

    write_mtz(paths["sf_cif"], paths["mtz"])
    mtz = read_mtz(paths["mtz"])
    free_label = free_flag_label(mtz)
    counts = free_flag_summary(mtz, free_label)
    if len(counts) < 2:
        raise FetchError(
            f"{pid}: free-flag column {free_label!r} holds the single value {list(counts)[0]} "
            f"for all {mtz.nreflections} reflections — no test set was deposited"
        )
    f_label, sig_label = amplitude_labels(mtz)
    return {
        **paths,
        "methods": methods,
        "free_label": free_label,
        "free_flag_counts": counts,
        "f_label": f_label,
        "sig_label": sig_label,
        "spacegroup": mtz.spacegroup_name,
        "cell": tuple(mtz.cell.parameters),
        "d_min": mtz.resolution_high(),
        "d_max": mtz.resolution_low(),
        "n_reflections": mtz.nreflections,
    }
