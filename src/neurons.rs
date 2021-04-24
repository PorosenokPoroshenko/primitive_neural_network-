use rand::Rng;

#[derive(Clone)]
struct Neuron {
    value: f64,
    bias: f64,
    connections: Vec<Connection>,
    activation_function: String,
}

impl Neuron {
    fn calculate(&self) -> f64 {
        // neuron = some activation function(sigmoid, tanh and etc.)(sum of all weights multiplyed by all inputs + bias)
        let mut sum: f64 = 0.0;
        for connection in self.connections.iter() {
            sum += connection.left_neuron.value as f64 * connection.weight as f64;
        }
        sum 
    }
}

#[derive(Clone)]
struct Connection {
    left_neuron: Neuron,
    weight: f64,
}

#[derive(Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

pub struct Network {
    layers: Vec<Layer>,

}

pub fn construct_network() -> Network{
        Network{
            layers: vec![]
        } 
    }
impl Network {

    pub fn fit_data(&mut self, data: Vec<f64>) {
        // for each data point create new
        // neuron, and push it in vector
        let mut layer = Layer{
                neurons: vec![]
            };
        for i in data.iter() {
            let neuron = Neuron {
                value: *i,
                bias: rand::thread_rng().gen_range(0.1..0.4),
                connections: vec![],
                activation_function: "tanh".to_string(),
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
    }
    pub fn get_output(&self)-> Vec<f64>{
        let mut output: Vec<f64> = vec![];
        for neuron in self.layers.last().unwrap().neurons.clone(){
            output.push(neuron.value);
        }; 
        output
    }

    pub fn create_layer(&mut self, num_of_neurons: i64) {
        let mut neurons = vec![];
        for _ in 0..num_of_neurons {
            //  create vector of connections between last layer neurons and current neuron
            let mut connections = vec![];
            let last_layer = self.layers.last().unwrap().neurons.clone();
            for neuron in last_layer.iter() {
                let connection = Connection {
                    left_neuron: neuron.clone(),
                    weight: rand::thread_rng().gen_range(0.1..0.5),
                };
                connections.push(connection);
            }

            let mut neuron = Neuron {
                value: 0.0,
                bias: rand::thread_rng().gen_range(0.1..0.5),
                connections,
                activation_function: "tanh".to_string(),
            };

            neuron.value = neuron.calculate().tanh();
            neurons.push(neuron);
        }

        let layer = Layer { neurons };
        self.layers.push(layer) } 
    // https://brilliant.org/wiki/backpropagation/#the-backpropagation-algorithm
    pub fn back_propogation(&self, target_values: Vec<f64>) {
        const LEARINING_RATE: f64 = 0.01;

        // let squared_error = |target: f64, output: f64| -> f64 { (target - output).sqrt() / 2.0 };

        // let mut errors: Vec<f64> = vec![];
        //for (i, neuron) in neurons.iter().enumerate() {
        //    let error = squared_error(target_values[i], neuron.value);
        //    errors.push(error);
        //}

        let mut total_gradient: Vec<f64> = vec![];

        let mut deltas: Vec<Vec<f64>> = vec![vec![]];
        for (i, layer) in self.layers.iter().rev().enumerate() {
            let neurons = layer.neurons.clone();
            let mut layer_deltas: Vec<f64> = Vec::new();

            for (j, neuron) in neurons.iter().enumerate() {
                // derevative of tanh
                let sigmoid = |x: f64| -> f64{
                    let e = std::f64::consts::E;
                    1.0/1.0+e.powf(-x)
                };

                let dsigmoid = |x: f64|->f64{
                    sigmoid(x) * (1.0-sigmoid(x))
                };

                let dtanh = |x: f64| -> f64 {
                    let e = std::f64::consts::E;
                    // sech^2(x)
                    4.0 / (e.powf(-x) + e.powf(x)).sqrt()
                };
                if i == 0 {
                    let delta: f64 =
                        (neuron.value - target_values[i]) * dsigmoid(neuron.calculate());
                    layer_deltas.push(delta);
                    println!("{} - delta in final layer", delta);

                } else {
                    fn burger(
                        mut sum: f64,
                        neuron: Neuron,
                        deltas: Vec<Vec<f64>>,
                        index: usize,
                    ) -> f64 {
                        for connection in neuron.connections.iter() {
                            for delta in deltas[deltas.len() - index].clone() {
                                sum += delta * connection.weight;
                            }
                        }
                        sum
                    }

                    let delta: f64 = dsigmoid(neuron.calculate())
                        * burger(0.0, neuron.clone(), deltas.clone(), i);
                    layer_deltas.push(delta);
                }

                deltas.push(layer_deltas.clone());
                for connection in neuron.connections.clone() {
                    // w_k_i_j: weight for node j in layer k for incoming node i
                    // o_k_j: output for neuron i in layer k r_k: number of nodes in layer k a_k_j: product sum plus bias for node i in layer k
                    // g: activation function
                    //
                    // d_k_j = dE/da_k_j

                    // in final layer
                    // dE/dw_m_i_j = d_m_j * o_m-1_i =
                    // = (output - target)g'(a_m_1)*o_m-1_i
                    // dE/db_m_i_j
                    // FOR FINAL LAYER
                    let final_gradient: f64 =
                        layer_deltas.clone()[j] * connection.left_neuron.value;
                    total_gradient.push(final_gradient);
                }
            }
        }
        // get average gradient
        let average_gradient: f64 =
            total_gradient.iter().fold(0.0, |sum, val| sum + val) / total_gradient.len() as f64;

        //update the weights
        for layer in self.layers.clone() {
            for neuron in layer.neurons.clone() {
                for mut connection in neuron.connections.clone() {
                    connection.weight += -LEARINING_RATE * average_gradient / connection.weight;
                }
            }
        }
    }
}
