func.func @matmul(%input : tensor<256x256xf32>, %weight : tensor<256x256xf32>) -> (tensor<256x256xf32>) {
    %z = tensor.empty(): tensor<256x256xf32>
    %output = linalg.matmul ins(%input, %weight: tensor<256x256xf32>, tensor<256x256xf32>) outs(%z: tensor<256x256xf32>) -> tensor<256x256xf32>
    func.return %output : tensor<256x256xf32>
}
