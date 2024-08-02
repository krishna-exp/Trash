# Trash

## Compile the kernels

```bash
> nvcc -arch=native <src>
> nsys profile -o test.nsys-prof ./a.out
> ncu --config-file off --kernel-name <kernel name> --export "report%i" --launch-count 1 --set full ./a.out
```

### NOTES

- These kernels are made without using warp intrinsics
- Throughput can be improved by folds with the intrinsics and co-operative thread groups
