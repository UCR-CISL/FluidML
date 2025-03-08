func.func @conv2d(%input : tensor<1x3x224x224xf32>, %filter : tensor<3x3x3x3xf32>) -> (tensor<1x3x222x222xf32>) {
    %c0f = arith.constant 0.0 : f32
    %z = tensor.empty(): tensor<1x3x222x222xf32>
    %0 = linalg.conv_2d_nchw_fchw ins(%input, %filter: tensor<1x3x224x224xf32>, tensor<3x3x3x3xf32>) outs(%z: tensor<1x3x222x222xf32>) -> tensor<1x3x222x222xf32>
    func.return %0 : tensor<1x3x222x222xf32>
}
