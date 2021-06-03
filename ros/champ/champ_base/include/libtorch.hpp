/**
 * @file libtorch.hpp
 * @author Nico Palomo (nicholas.j.palomo@jpl.nasa.gov)
 * @brief Static library for running inferencing in Libtorch and for converting
 * between Eigen and LibTorch
 * @version 0.1
 * @date 2021-02-26
 *
 */

#include "stdio.h"
#include <Eigen/Core>
#include <torch/script.h> ///< One-stop header

namespace libtorch {

class libtorch {
public:
  libtorch(void) {}

  void load(std::string dir);

  template <typename V, int Dim1, int Dim2>
  void forward(Eigen::Matrix<V, Dim1, 1>& M, Eigen::Matrix<V, Dim2, 1>& N);

  template <typename V, int Dim>
  void eigen2libtorch(Eigen::Matrix<V, Dim, 1>& M,
                      std::vector<torch::jit::IValue>& T);

  template <typename V, int Dim>
  void libtorch2eigen(at::Tensor& Tin, Eigen::Matrix<V, Dim, 1>& M);

private:
  torch::jit::script::Module module_;
};

} // namespace libtorch

/**
 * @brief Load the network parameters to a network module
 *
 * @param torchscriptNetDir
 */
void libtorch::libtorch::load(std::string torchscriptNetDir) {
  try {
    std::cout << "[libtorch] Attempting to load network parameters from ";

    std::cout << torchscriptNetDir << std::endl;

    module_ = torch::jit::load(torchscriptNetDir);
    module_.to(torch::kCPU);
    module_.eval();
    torch::NoGradGuard no_grad_;

    std::cout << "[libtorch] Student network parameters loaded" << std::endl;

  } catch (const c10::Error& e) {
    std::cerr << "Error loading control module!\n";
  }
}

/**
 * @brief Evaluate the network
 *
 * @tparam V
 * @tparam Dim1 - action dimension
 * @tparam Dim2 - state dimension
 * @param M - action vector
 * @param N - state vector
 */
template <class V, int Dim1, int Dim2>
void libtorch::libtorch::forward(Eigen::Matrix<V, Dim1, 1>& M,
                                 Eigen::Matrix<V, Dim2, 1>& N) {
  M.setZero(Dim1);

  std::vector<torch::jit::IValue> input;
  input.push_back(torch::ones({Dim2}));

  eigen2libtorch<V, Dim2>(N, input);

  at::Tensor Mtensor = module_.forward(input).toTensor();

  libtorch2eigen<V, Dim1>(Mtensor, M);
}

/**
 * @brief Convert Eigen matrix to libtorch tensor
 *
 * @tparam V
 * @tparam Dim
 * @param M
 * @param T
 */
template <typename V, int Dim>
void libtorch::libtorch::eigen2libtorch(Eigen::Matrix<V, Dim, 1>& M,
                                        std::vector<torch::jit::IValue>& T) {
  std::vector<V> vec(M.data(), M.data() + M.rows());
  T[0] = torch::tensor(vec).clone();
}

/**
 * @brief Convert libtorch tensor to Eigen matrix
 *
 * @tparam V
 * @tparam Dim
 * @param Tin
 * @param M
 */
template <typename V, int Dim>
void libtorch::libtorch::libtorch2eigen(at::Tensor& Tin,
                                        Eigen::Matrix<V, Dim, 1>& M) {
  for (int i = 0; i < Dim; i++)
    M[i] = static_cast<V>(Tin.to(torch::kCPU).data_ptr<float>()[i]);
}