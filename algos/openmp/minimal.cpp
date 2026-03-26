#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

float squared_sum(const std::vector<float>& values) {
    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        sum += values[i] * values[i];
    }
    return sum;
}

void scale_inplace(std::vector<float>& values, float scale) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] *= scale;
    }
}

int main() {
    std::vector<float> values(16);
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        values[i] = static_cast<float>(i + 1);
    }

#ifdef _OPENMP
    std::cout << "max threads: " << omp_get_max_threads() << '\n';
#else
    std::cout << "OpenMP not enabled\n";
#endif

    std::cout << "squared sum: " << squared_sum(values) << '\n';
    scale_inplace(values, 0.5f);

    std::cout << "scaled values:";
    for (float value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
    return 0;
}
