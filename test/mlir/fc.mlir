func.func @fc(%input : tensor<256x256xf32>, %weight : tensor<256x256xf32>) -> (tensor<256x256xf32>) {
    %z = tensor.empty(): tensor<256x256xf32>
    %0 = linalg.matmul ins(%input, %weight: tensor<256x256xf32>, tensor<256x256xf32>) outs(%z: tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = linalg.softmax dimension(1) ins(%0 : tensor<256x256xf32>) outs(%z : tensor<256x256xf32>) -> tensor<256x256xf32>
    func.return %1 : tensor<256x256xf32>
}
