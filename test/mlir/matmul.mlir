func.func @matmul(%input : tensor<4096x4096xf32>, %weight : tensor<4096x4096xf32>) -> (tensor<4096x4096xf32>) {
    %z = tensor.empty(): tensor<4096x4096xf32>
    %output = linalg.matmul ins(%input, %weight: tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%z: tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    func.return %output : tensor<4096x4096xf32>
}
