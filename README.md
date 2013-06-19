HoneyComb
=========

### Data Parallel Bytecode for Simpler GPU Programming


Examples:
---------

Adding two vectors:

```
  n = DIM a0 0
  PARFOR n
    offset = MUL_SCALAR idx.x vecsize
    v0 = LOAD a0 offset 
    v1 = LOAD a1 offset
    IADD v1 v0 
    STORE v1 a1 s0 
```

Naive matrix multiply: 

```
  n = DIM a0 0 
  m = DIM a1 1
  PARFOR n, m  
    k = DIM a0 1 
    FOR_OFFSETS(k, offset)   
      v0 = LOAD_ROW a0 idx.x offset
      v1 = LOAD_COL a1 idx.y offset
      IMUL v1 v0 
      total = SUM v1 
      STORE_SCALAR_2D total a2 idx.x idx.y 
```

Tree-structured parallel sum of a 1D vector using an outermost loop:

```
  b = CONST 1
  WHILE b
    n_full = DIM a0 0 
    n_small = DIV n vecsize 
    a1 = ALLOC n_small
    PARFOR n_full BY vecsize  
      v0 = LOAD a0 idx.x 
      total = SUM v0 
      i = DIV_SCALAR idx.x vecsize 
      STORE_SCALAR total a1 i 
    b = GT n_small 1
    a0 = a1 
  res =  LOAD_SCALAR a0 0 
  RETURN res 
```

Or, more pleasantly, using a global reduce: 


```
  n = DIM a0 0
  REDUCE n
    v0 = LOAD a0 idx.x 
    total = SUM v0 
    return total 
```


  
  
