#include <iostream>
#include <vector>
#include <ctime>

using Matrix = std::vector<std::vector<int>>;

Matrix ReadMatrix(int N) {
  Matrix m;
  m.resize(N);
  for(int r = 0; r < N; ++r) {
    m[r].resize(N);
    for(int c = 0; c < N; ++c) {
      std::cin >> m[r][c];
    }
  }
  return m;
}

Matrix SimpleMult(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res;
  res.resize(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
    for(int c = 0; c < s; ++c) {
      for(int x = 0; x < s; ++x) {
        res[r][c] += left[R1 + r][C1 + x] * right[R2 + x][C2 + c];
      }
    }
  }
  return res;
}

Matrix CachedMult(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res;
  res.resize(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
  }
  for(int c = 0; c < s; ++c) {
    std::vector<int> cache(s);
    for(int r = 0; r < s; ++r) cache[r] = right[R2 + r][C2 + c];
    for(int r = 0; r < s; ++r) {
      for(int x = 0; x < s; ++x) {
        res[r][c] += left[R1 + r][C1 + x] * cache[x];
      }
    }
  }
  return res;
}

Matrix Add(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
    for(int c = 0; c < s; ++c) {
      res[r][c] = left[R1 + r][C1 + c] + right[R2 + r][C2 + c];
    }
  }
  return res;
}

Matrix Sub(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
    for(int c = 0; c < s; ++c) {
      res[r][c] = left[R1 + r][C1 + c] - right[R2 + r][C2 + c];
    }
  }
  return res;
}

void SetMatrix(Matrix &out, Matrix in, int R, int C, int s) {
  for(int r = 0; r < s; ++r) {
    for(int c = 0; c < s; ++c) {
      out[R + r][C + c] = in[r][c];
    }
  }
}

Matrix SmartMult(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  if(s <= 64) {
    return CachedMult(left, right, R1, C1, R2, C2, s);
  }
  Matrix res(s);
  for(int r = 0; r < s; ++r) res[r].resize(s);

  int s2 = s / 2;

  Matrix P1r = Sub(right, right, R2, C2 + s2, R2 + s2, C2 + s2, s2);
  Matrix P1 = SmartMult(left, P1r, R1, C1, 0, 0, s2);

  Matrix P2l = Add(left, left, R1, C1, R1, C1 + s2, s2);
  Matrix P2 = SmartMult(P2l, right, 0, 0, R2 + s2, C2 + s2, s2);

  Matrix P3l = Add(left, left, R1 + s2, C1, R1 + s2, C1 + s2, s2);
  Matrix P3 = SmartMult(P3l, right, 0, 0, R2, C2, s2);

  Matrix P4r = Sub(right, right, R2 + s2, C2, R2, C2, s2);
  Matrix P4 = SmartMult(left, P4r, R1 + s2, C1 + s2, 0, 0, s2);

  Matrix P5l = Add(left, left, R1, C1, R1 + s2, C1 + s2, s2);
  Matrix P5r = Add(right, right, R2, C2, R2 + s2, C2 + s2, s2);
  Matrix P5 = SmartMult(P5l, P5r, 0, 0, 0, 0, s2);

  Matrix P6l = Sub(left, left, R1, C1 + s2, R1 + s2, C1 + s2, s2);
  Matrix P6r = Add(right, right, R2 + s2, C2, R2 + s2, C2 + s2, s2);
  Matrix P6 = SmartMult(P6l, P6r, 0, 0, 0, 0, s2);

  Matrix P7l = Sub(left, left, R1, C1, R1 + s2, C1, s2);
  Matrix P7r = Add(right, right, R2, C2, R2, C2 + s2, s2);
  Matrix P7 = SmartMult(P7l, P7r, 0, 0, 0, 0, s2);

  Matrix left1 = Add(P5, P4, 0, 0, 0, 0, s2);
  Matrix right1 = Sub(P2, P6, 0, 0, 0, 0, s2);
  SetMatrix(res, Sub(left1, right1, 0, 0, 0, 0, s2), 0, 0, s2);

  SetMatrix(res, Add(P1, P2, 0, 0, 0, 0, s2), 0, s2, s2);

  SetMatrix(res, Add(P3, P4, 0, 0, 0, 0, s2), s2, 0, s2);

  left1 = Add(P5, P1, 0, 0, 0, 0, s2);
  right1 = Add(P3, P7, 0, 0, 0, 0, s2);
  SetMatrix(res, Sub(left1, right1, 0, 0, 0, 0, s2), s2, s2, s2);

  return res;
}

bool CompareMatrix(Matrix &left, Matrix &right) {
  int N = left.size();
  for(int r = 0; r < N; ++r) {
    for(int c = 0; c < N; ++c) {
      if(left[r][c] != right[r][c]) return false;
    }
  }
  return true;
}

int main() {
  int N;
  std::cin >> N;
  time_t start = std::time(nullptr);
  Matrix m1 = ReadMatrix(N);
  Matrix m2 = ReadMatrix(N);
  time_t end = std::time(nullptr);
  std::cout << "Read time: " << (end - start) << " seconds.\n";
  start = std::time(nullptr);
  Matrix res1 = SimpleMult(m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Simple time: " << (end - start) << " seconds.\n";
  start = std::time(nullptr);
  Matrix res2 = CachedMult(m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Cached time: " << (end - start) << " seconds.\n";
  std::cout << (CompareMatrix(res1, res2) ? "Cached works!\n" : "Chached failed!\n");
  start = std::time(nullptr);
  Matrix res3 = SmartMult(m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Smart time: " << (end - start) << " seconds.\n";
  std::cout << (CompareMatrix(res1, res3) ? "Smart works!\n" : "Smart failed!\n");
}
