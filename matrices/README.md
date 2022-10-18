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
g++ --std=c++17 main.cc
```

## Test solver

```
cat s128.txt | ./a.out
```