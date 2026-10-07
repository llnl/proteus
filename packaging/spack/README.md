# Proteus Spack Package

## Installation

Add this repository:
```bash
git clone https://github.com/LLNL/proteus.git
spack repo add proteus/packaging/spack/spack_repo/proteus
```

Install the latest main branch:
```bash
spack install proteus@main
```

Install with CUDA or HIP support:
```bash
spack install proteus@main +cuda
spack install proteus@main +hip
```
