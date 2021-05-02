mod neurons;

fn main() {
    let data = vec![1.0, 1.0, 1.0];
    let target_values = vec![0.0];

    
    let mut network = neurons::construct_network();
    // 3 layers 
    network.fit_data(data);
    network.create_layer(2);
    network.create_layer(1);

    for _ in 0..15{
        network.back_propogation(target_values.clone());
        network.forward_propogation(&neurons::sigmoid);
    }
    let output = network.get_output();

    network.display_network();
    println!("{:?}", output);
}
