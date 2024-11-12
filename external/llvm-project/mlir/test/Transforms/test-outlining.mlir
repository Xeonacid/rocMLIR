// CHECK-LABEL: func private @test_fusion4__part_0
// CHECK: tosa.conv2d
// CHECK: tosa.abs
// +++pf:  This test used to absorb the tosa.add, too, but doesn't now.
// CHECK: return
// CHECK: func @test_fusion4
// CHECK: call @test_fusion4__part_0
func.func @test_fusion4(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.add"(%0, %1) {} : (tensor<128x128x30x30xf32>, tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %2 : tensor<128x128x30x30xf32>
}


#map0 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @resnet50(%arg0: memref<1x32x32x64xf32>, %arg1: memref<64x3x3x64xf32>, %arg2: memref<1x32x32x64xf32>) {
  %cst = arith.constant 6.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() {alignment = 128 : i64} : memref<1x34x34x64xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<1x34x34x64xf32>)
  %1 = memref.subview %0[0, 1, 1, 0] [1, 32, 32, 64] [1, 1, 1, 1] : memref<1x34x34x64xf32> to memref<1x32x32x64xf32, strided<[73984, 2176, 64, 1], offset: 2240>>
  memref.copy %arg0, %1 : memref<1x32x32x64xf32> to memref<1x32x32x64xf32, strided<[73984, 2176, 64, 1], offset: 2240>>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<3x3x64x64xf32>
  linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : memref<64x3x3x64xf32>) outs(%2 : memref<3x3x64x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
  }
  %3 = memref.alloc() {alignment = 128 : i64} : memref<1x32x32x64xf32>
  linalg.fill ins(%cst_0 : f32) outs(%3 : memref<1x32x32x64xf32>)
  linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %2 : memref<1x34x34x64xf32>, memref<3x3x64x64xf32>) outs(%3 : memref<1x32x32x64xf32>)
  linalg.add ins(%3, %arg2 : memref<1x32x32x64xf32>, memref<1x32x32x64xf32>)
       outs(%arg2 : memref<1x32x32x64xf32>)
  memref.dealloc %0 : memref<1x34x34x64xf32>
  memref.dealloc %2 : memref<3x3x64x64xf32>
  memref.dealloc %3 : memref<1x32x32x64xf32>
  return
}
