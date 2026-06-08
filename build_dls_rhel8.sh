#!/bin/bash
# ============================================================================
# build_dls_rhel8.sh -- build this RELION (ver5.1-aretomo3-support) on DLS RHEL 8
# ============================================================================
# Builds the C++/CUDA RELION (incl. the GUI with the new AreTomo3 tab and the
# relion_align_tiltseries AreTomo3 backend) into a writable /scratch prefix.
# No sudo; everything via the DLS module system.
#
# RHEL 8 constraints honoured (see RHEL_ENVIRONMENT_DETAILS.md in the parent
# automation repo):
#   * HOME directory quota cannot be used for build artefacts/caches -> TMPDIR,
#     CUDA cache, and the build/install trees all live under scratch, never $HOME.
#   * The default RHEL 8 GCC (8.5) is too old and nvcc is not on PATH by default
#     -> a gcc and a cuda module are loaded explicitly.
#
# USAGE (on a DLS RHEL 8 host with the module system):
#   ./build_dls_rhel8.sh
# Override anything via env vars, e.g.:
#   CUDA_ARCH=86 RELION_PREFIX=/scratch/$USER/relion-aretomo3 \
#     MOD_CUDA=cuda/12.8 ./build_dls_rhel8.sh
# Check `module avail cuda gcc cmake openmpi` and adjust MOD_* if a version is gone.
# ============================================================================
set -euo pipefail

# ---- source tree (this repo) -----------------------------------------------
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- configurable knobs (env-overridable) ----------------------------------
# RELION_PREFIX = the INSTALL location (CMAKE_INSTALL_PREFIX): where `make install`
# puts the finished, runnable RELION (bin/, lib/) -- this is what you add to PATH.
# On DLS, /scratch/$USER is writable and survives reboots (per RHEL_ENVIRONMENT_DETAILS.md);
# /dls_sw is read-only and $HOME has a quota, so neither can be used.
RELION_PREFIX="${RELION_PREFIX:-/scratch/${USER}/relion-5.1-aretomo3}"
# RELION_BUILD = the compile working dir (object files / cmake cache); a subdir of
# the prefix. Only needed after install for incremental rebuilds.
RELION_BUILD="${RELION_BUILD:-${RELION_PREFIX}/build}"
# GPU compute capability: 89=Ada (RTX 5000 Ada / 4090 / L40), 86=A5000/3090,
# 80=A100, 90=H100. ws592 (the project's GPU) is Ada -> 89.
CUDA_ARCH="${CUDA_ARCH:-89}"
NPROC="${NPROC:-$(nproc)}"

# Module versions are auto-detected from `module avail` below (newest compatible).
# Override any of MOD_GCC / MOD_CUDA / MOD_CMAKE / MOD_MPI to force a specific one.

# ---- RHEL 8 constraint: redirect caches/tmp OFF $HOME onto scratch ----------
SCRATCH="${SCRATCH:-/scratch/${USER}}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$SCRATCH/.nv}"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"

# ---- module system ---------------------------------------------------------
for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
    [ -r "$init" ] && . "$init" 2>/dev/null || true
done
if ! command -v module >/dev/null 2>&1 && ! type module >/dev/null 2>&1; then
    echo "ERROR: no 'module' command -- run this on a DLS RHEL 8 host." >&2; exit 1
fi

# Tiny `module avail` probe: newest available "<base>/<prefer>..." module, else fallback.
# The version filters keep choices RELION/CUDA-compatible (cuda 12.x, gcc <=13, cmake 3.x).
latest_module() {   # latest_module <base> <prefer-regex> <fallback>
    local hit
    hit="$(module -t avail "$1" 2>&1 \
            | sed 's/(default)//; s/[[:space:]]*$//' \
            | grep -E "^$1/$2[0-9.]*$" | sort -V | tail -1)"
    [ -n "$hit" ] && printf '%s' "$hit" || printf '%s' "$3"
}
MOD_GCC="${MOD_GCC:-$(latest_module gcc     '1[0-3]\.' gcc/12)}"        # <=13: CUDA host-compiler compat
MOD_CUDA="${MOD_CUDA:-$(latest_module cuda  '12\.'     cuda/12.6)}"     # stay on 12.x for RELION
MOD_CMAKE="${MOD_CMAKE:-$(latest_module cmake '3\.'    cmake/3.31.6)}"  # avoid cmake 4.x
MOD_MPI="${MOD_MPI:-$(latest_module openmpi '4\.'      openmpi/4.1.6)}"

module purge 2>/dev/null || true
echo "Loading modules (auto-picked; override with MOD_*): $MOD_GCC $MOD_CUDA $MOD_CMAKE $MOD_MPI"
module load "$MOD_GCC" "$MOD_CUDA" "$MOD_CMAKE" "$MOD_MPI"

# ---- preflight -------------------------------------------------------------
fail=0
command -v nvcc  >/dev/null || { echo "ERROR: nvcc not on PATH (cuda module?)"  >&2; fail=1; }
command -v cmake >/dev/null || { echo "ERROR: cmake not on PATH"                 >&2; fail=1; }
command -v g++   >/dev/null || { echo "ERROR: g++ not on PATH (gcc module?)"     >&2; fail=1; }
command -v mpicxx>/dev/null || echo "WARNING: mpicxx not found -- MPI variants (relion_*_mpi) may be skipped." >&2
[ "$fail" -eq 0 ] || exit 1

echo "============================================================"
echo "  RELION build (DLS RHEL 8)"
echo "============================================================"
echo "  source     : $SRC  ($(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$SRC" describe --tags 2>/dev/null || echo '?'))"
echo "  build      : $RELION_BUILD"
echo "  install    : $RELION_PREFIX"
echo "  CUDA_ARCH  : $CUDA_ARCH"
echo "  gcc        : $(g++ --version | head -1)"
echo "  nvcc       : $(nvcc --version | grep -i release | sed 's/^ *//')"
echo "  cmake      : $(cmake --version | head -1)"
echo "  TMPDIR     : $TMPDIR   (kept off \$HOME per RHEL_ENVIRONMENT_DETAILS.md)"
echo "============================================================"

# ---- configure -------------------------------------------------------------
mkdir -p "$RELION_BUILD"
cd "$RELION_BUILD"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$RELION_PREFIX" \
    -DCUDA=ON \
    -DCUDA_ARCH="$CUDA_ARCH" \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
    "$SRC"

# ---- build + install -------------------------------------------------------
make -j "$NPROC"
make install

echo
echo "============================================================"
echo "  DONE -- installed to $RELION_PREFIX"
echo "============================================================"
cat <<EOF
To use this build:
  export PATH=$RELION_PREFIX/bin:\$PATH
  # point the new AreTomo3 tab at the AreTomo3 binary:
  export RELION_ARETOMO3_EXECUTABLE=/dls_sw/apps/EM/aretomo3/2.2.8/AreTomo3/AreTomo3
  # (existing AreTomo2 / batchruntomo defaults still work for the other tabs)
  relion    # AlignTiltSeries job should now show an "AreTomo3" tab

Notes:
  * GUI build needs X11/OpenGL dev libs for the bundled FLTK. If the GUI target
    fails, load 'module load fltk/1.3.4' (or install the X11 -devel deps) and
    re-run, or build headless with -DGUI=OFF (no AreTomo3 tab then).
  * The python tomo programs need a RELION conda env at RUNTIME (not for this
    build). Reuse the DLS one:
      export RELION_PYTHON_EXECUTABLE=/dls_sw/apps/EM/relion/5.0/conda/bin/python
  * Re-running this script does an incremental 'make' in $RELION_BUILD.
EOF
