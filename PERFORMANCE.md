# Performance notes for this branch

## Start-up defaults

On the GPU, `relion_refine` and `relion_refine_mpi` choose the values of this branch's performance switches when they
start, from the run type (ab initio, 2D classification, 3D classification or auto-refine) and the GPU (the lowest
compute capability among the visible devices), and print the configuration they chose in the log. A switch you set in the
environment always wins, and `RELION_PERF_DEFAULTS=0` turns the choice off (every switch off, as in upstream RELION).
The rules are in `src/perf_defaults.h`.

## Refinement under NVIDIA MPS

When several MPI workers share one GPU, as in the usual 3D auto-refinement layout (two workers per GPU, one for each
half-set), each worker has its own CUDA context and the GPU runs them in turns. NVIDIA's Multi-Process Service (MPS)
lets their kernels run at the same time. It needs no change to RELION and no administrator: a job starts its own MPS
daemon for its GPUs and stops it when it ends.

`relion_with_mps` (installed with the other scripts, source `scripts/with_mps`) does that around any command:

```bash
relion_with_mps -- mpirun -np 3 -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_VISIBLE_DEVICES \
    relion_refine_mpi --auto_refine --split_random_halves ... --gpu 0 --j 4
```

- It serves the GPUs in `CUDA_VISIBLE_DEVICES` (or `--gpus <ids>`) and renumbers them from 0 for the command, so
  address them as `--gpu 0` (or `--gpu 0:1` for two).
- With OpenMPI, pass `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_VISIBLE_DEVICES` to the ranks (`-x`, as above); otherwise the
  ranks start their own contexts and nothing changes.
- When the command ends it prints how many client processes the daemon served: the number of ranks that used the
  GPU (3 for the example) means it worked, 0 means the ranks did not see the daemon.
- In a queue submission template, wrap the launch line, e.g.
  `relion_with_mps -- mpirun -n XXXmpinodesXXX -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_VISIBLE_DEVICES XXXcommandXXX`.
- It needs `nvidia-cuda-mps-control` on the node and a GPU in compute mode Default or Exclusive_Process, and is meant
  for GPUs allocated to the one job.

**Measured** (3D auto-refinement of an 11,599-particle EMPIAR-10017 subset to convergence, 2 workers × 4 threads per
GPU, this branch at its start-up defaults; whole nodes, every GPU running the same configuration, 8 or 16 runs each):

| GPU | without MPS | with MPS | speed-up |
|---|---:|---:|---:|
| Tesla V100S | 532 s | 331 s | 1.61× |
| RTX PRO 6000 Blackwell | 426 s | 154 s | 2.78× |

Every run took the same number of iterations to the same resolution, and the final FSC curves differ from those
without MPS no more than repeated runs without MPS differ from each other.

- **Keep two workers per GPU.** Four workers × 2 threads under MPS were a little faster (1.70× and 3.00×), but the
  refinement then converged in a varying number of iterations (16–21 instead of 21), with final FSC curves that
  differ more (no better or worse on average). Eight workers × 1 thread were no faster than four.
- **MPS shortens only the GPU's part of each iteration** (the expectation step). The CPU-side steps between
  iterations stay as long as before, so they become a quarter to a third of the run.
- **One process gains nothing:** the threads of one process already share one context. Classification with one
  worker per GPU, ab initio (a single process) and the CPU-only programs are unaffected.
- **GPU memory is still divided between the workers**, so larger boxes may limit how many fit on one GPU.
