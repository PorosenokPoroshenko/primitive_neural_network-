mod neurons;

fn create_data(){
}
fn main() {
    let data = vec![vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]];
    let test_data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let target_values = vec![vec![0.0], vec![1.0]];
    
    let mut network = neurons::construct_network();
    network.fit_data(data);
    network.create_layer(1);

    for _ in 0..30{
        network.back_propogation(target_values.clone());
    }
    let output = network.get_output();

    println!("{:?}", output);
}
