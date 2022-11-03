# Compile generator

```
g++ --std=c++17 gen.cc
```

## Generate test file

Might not work :-/

```
echo 128 | ./a.out > s128.txt
```

# Compile solver

```
g++ -march=native -mtune=native -O3 -std=c++17 main.cc
```

```
zig build-exe -O ReleaseFast main.zig
```

## Test solver

```
cat s128.txt | ./a.out
```

```
cat s128.txt | ./main
```
