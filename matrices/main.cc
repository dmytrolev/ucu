#include <iostream>
#include <vector>
#include <ctime>
#include <cstring>
#include <smmintrin.h>

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

Matrix CachedMultWithSSE(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res;
  res.resize(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
  }
  for(int c = 0; c < s; ++c) {
    __m128i R[s/4];
    for(int r = 0; r < s; r += 4) {
      R[r/4] = _mm_set_epi32(right[R2 + r][C2 + c], right[R2 + r + 1][C2 + c], right[R2 + r + 2][C2 + c], right[R2 + r + 3][C2 + c]);
    }
    for(int r = 0; r < s; ++r) {
      __m128i V = _mm_setzero_si128();
      for(int x = 0; x < s; x += 4) {
        __m128i L = _mm_set_epi32(left[R1 + r][C1 + x], left[R1 + r][C1 + x + 1], left[R1 + r][C1 + x + 2], left[R1 + r][C1 + x + 3]);
        __m128i V1 = _mm_mullo_epi32(L, R[x/4]);
        V = _mm_add_epi32(V1, V);
      }
      res[r][c] = _mm_extract_epi32(V, 0) + _mm_extract_epi32(V, 1) + _mm_extract_epi32(V, 2) + _mm_extract_epi32(V, 3);
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

Matrix AddWithSSE(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
    for(int c = 0; c < s; c += 4) {
      __m128i L = _mm_set_epi32(left[R1 + r][C1 + c], left[R1 + r][C1 + c + 1], left[R1 + r][C1 + c + 2], left[R1 + r][C1 + c + 3]);
      __m128i R = _mm_set_epi32(right[R2 + r][C2 + c], right[R2 + r][C2 + c + 1], right[R2 + r][C2 + c + 2], right[R2 + r][C2 + c + 3]);
      __m128i V = _mm_add_epi32(L, R);
      res[r][c] = _mm_extract_epi32(V, 3);
      res[r][c + 1] = _mm_extract_epi32(V, 2);
      res[r][c + 2] = _mm_extract_epi32(V, 1);
      res[r][c + 3] = _mm_extract_epi32(V, 0);
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

Matrix SubWithSSE(Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  Matrix res(s);
  for(int r = 0; r < s; ++r) {
    res[r].resize(s);
    for(int c = 0; c < s; c += 4) {
      __m128i L = _mm_set_epi32(left[R1 + r][C1 + c], left[R1 + r][C1 + c + 1], left[R1 + r][C1 + c + 2], left[R1 + r][C1 + c + 3]);
      __m128i R = _mm_set_epi32(right[R2 + r][C2 + c], right[R2 + r][C2 + c + 1], right[R2 + r][C2 + c + 2], right[R2 + r][C2 + c + 3]);
      __m128i V = _mm_sub_epi32(L, R);
      res[r][c] = _mm_extract_epi32(V, 3);
      res[r][c + 1] = _mm_extract_epi32(V, 2);
      res[r][c + 2] = _mm_extract_epi32(V, 1);
      res[r][c + 3] = _mm_extract_epi32(V, 0);
    }
  }
  return res;
}

void SetMatrix(Matrix &out, Matrix in, int R, int C, int s) {
  for(int r = 0; r < s; ++r) {
    std::copy(in[r].begin(), in[r].end(), out[R + r].begin() + C);
  }
}

template<typename Mult, typename Add, typename Sub>
Matrix SmartMult(Mult mult, Add add, Sub sub, Matrix &left, Matrix &right, int R1, int C1, int R2, int C2, int s) {
  if(s <= 64) {
    return mult(left, right, R1, C1, R2, C2, s);
  }
  Matrix res(s);
  for(int r = 0; r < s; ++r) res[r].resize(s);

  int s2 = s / 2;

  Matrix P1r = sub(right, right, R2, C2 + s2, R2 + s2, C2 + s2, s2);
  Matrix P1 = SmartMult(mult, add, sub,  left, P1r, R1, C1, 0, 0, s2);

  Matrix P2l = add(left, left, R1, C1, R1, C1 + s2, s2);
  Matrix P2 = SmartMult(mult, add, sub,  P2l, right, 0, 0, R2 + s2, C2 + s2, s2);

  Matrix P3l = add(left, left, R1 + s2, C1, R1 + s2, C1 + s2, s2);
  Matrix P3 = SmartMult(mult, add, sub,  P3l, right, 0, 0, R2, C2, s2);

  Matrix P4r = sub(right, right, R2 + s2, C2, R2, C2, s2);
  Matrix P4 = SmartMult(mult, add, sub,  left, P4r, R1 + s2, C1 + s2, 0, 0, s2);

  Matrix P5l = add(left, left, R1, C1, R1 + s2, C1 + s2, s2);
  Matrix P5r = add(right, right, R2, C2, R2 + s2, C2 + s2, s2);
  Matrix P5 = SmartMult(mult, add, sub,  P5l, P5r, 0, 0, 0, 0, s2);

  Matrix P6l = sub(left, left, R1, C1 + s2, R1 + s2, C1 + s2, s2);
  Matrix P6r = add(right, right, R2 + s2, C2, R2 + s2, C2 + s2, s2);
  Matrix P6 = SmartMult(mult, add, sub,  P6l, P6r, 0, 0, 0, 0, s2);

  Matrix P7l = sub(left, left, R1, C1, R1 + s2, C1, s2);
  Matrix P7r = add(right, right, R2, C2, R2, C2 + s2, s2);
  Matrix P7 = SmartMult(mult, add, sub,  P7l, P7r, 0, 0, 0, 0, s2);

  Matrix left1 = add(P5, P4, 0, 0, 0, 0, s2);
  Matrix right1 = sub(P2, P6, 0, 0, 0, 0, s2);
  SetMatrix(res, sub(left1, right1, 0, 0, 0, 0, s2), 0, 0, s2);

  SetMatrix(res, add(P1, P2, 0, 0, 0, 0, s2), 0, s2, s2);

  SetMatrix(res, add(P3, P4, 0, 0, 0, 0, s2), s2, 0, s2);

  left1 = add(P5, P1, 0, 0, 0, 0, s2);
  right1 = add(P3, P7, 0, 0, 0, 0, s2);
  SetMatrix(res, sub(left1, right1, 0, 0, 0, 0, s2), s2, s2, s2);

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

  start = std::time(nullptr);
  Matrix res3 = CachedMultWithSSE(m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Cached with SSE time: " << (end - start) << " seconds.\n";

  start = std::time(nullptr);
  Matrix res4 = SmartMult(SimpleMult, Add, Sub, m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Smart time with simple mult: " << (end - start) << " seconds.\n";

  start = std::time(nullptr);
  Matrix res5 = SmartMult(CachedMult, Add, Sub, m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Smart time with cached mult: " << (end - start) << " seconds.\n";

  start = std::time(nullptr);
  Matrix res6 = SmartMult(CachedMultWithSSE, Add, Sub, m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Smart time with Cached Mult+SSE: " << (end - start) << " seconds.\n";

  start = std::time(nullptr);
  Matrix res7 = SmartMult(CachedMultWithSSE, AddWithSSE, SubWithSSE, m1, m2, 0, 0, 0, 0, N);
  end = std::time(nullptr);
  std::cout << "Smart time with Mult+SSE and Add/Sub+SSE: " << (end - start) << " seconds.\n";

  // std::cout << (CompareMatrix(res2, res4) ? "SSE work!\n" : "SSE failed!\n");
}
