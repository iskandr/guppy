HoneyComb
=========

## Data Parallel Bytecode for Simpler GPU Programming


Examples:
---------

Adding two vectors:

```
  s0 = DIM a0 0
  PARFOR s0
    s0 = MUL_SCALAR idx.x vecsize
    v0 = LOAD a0 s0 
    v1 = LOAD a1 s0 
    IADD v1 v0 
    STORE v1 a1 s0 
```

Naive matrix multiply: 

```
  s0 = DIM a0 0 
  s1 = DIM a1 1
  PARFOR s0, s1 
    s0 = DIM a0 1 
    FOR_OFFSETS s0 s1
      v0 = LOAD_ROW a0 idx.x s1
      v1 = LOAD_COL a1 idx.y s1 
      IMUL v1 v0 
      f0 = SUM v1 
      STORE_SCALAR_2D f0 a2 idx.x idx.y 
```
