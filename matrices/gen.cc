#include <cstdlib>
#include <ctime>
#include <iostream>

void PrintRandomMatrix(int N) {
  for(int r = 0; r < N; ++r) {
    for(int c = 0; c < N; ++c) {
      std::cout << (int)(std::rand()/((RAND_MAX + 1u)/10)) - 4 << (c == N - 1 ? "" : " ");
    }
    std::cout << std::endl;
  }
}

int main() {
  std::srand(std::time(nullptr));
  int N;
  std::cin >> N;
  std::cout << N << std::endl;
  PrintRandomMatrix(N);
  std::cout << std::endl;
  PrintRandomMatrix(N);
}
