use burn::backend::Wgpu;
use burn::tensor::Tensor;

//Type alias for the backend to use.
type Backend = Wgpu;

fn main() {
    //creation of two tensors, the first with
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
