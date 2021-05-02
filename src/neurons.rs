use rand::Rng;

#[derive(Clone)]
struct Neuron {
    value: f64,
    index: i64,
    layer_ind: i64,
    is_input: bool,
    connections: Vec<Connection>,
    activation_function: String,
}
impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Neuron {
    fn calculate(&self) -> f64 {
        // value of neuron = some activation function(sigmoid, tanh and etc.)(sum of all weights multiplyed by all inputs + bias)
        let mut sum: f64 = 0.0;
        for connection in self.connections.iter() {
            sum += connection.left_neuron.value as f64 * connection.weight as f64 + 1.0;
        }
        sum
    }
    fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}

#[derive(Clone)]
struct Connection {
    left_neuron: Neuron,
    right_neuron: Option<Neuron>,
    weight: f64,
}

impl Connection {
    fn set_weight(&mut self, value: f64) {
        self.weight = value;
    }
}

#[derive(Clone)]
struct Layer {
    neurons: Vec<Neuron>,
    bias: f64,
    index: i64,
}

pub struct Network {
    layers: Vec<Layer>,
    activation_fn: String,
    number_of_neurons: i64,
    number_of_layers: i64,
}

pub fn construct_network() -> Network {
    Network {
        layers: vec![],
        activation_fn: "sigmoid".to_string(),
        number_of_neurons: 0,
        number_of_layers: 0,
    }
}

// one of the possible activation functions
fn tanh(x: f64) -> f64 {
    let e = std::f64::consts::E;
    // sech^2(x)
    4.0 / (e.powf(-x) + e.powf(x)).sqrt()
}

// one of the possible activation functions
pub fn sigmoid(x: f64) -> f64 {
    let e = std::f64::consts::E;
    1.0 / 1.0 + e.powf(-x)
}

// derevative of tanh function
pub fn dtanh(x: f64) -> f64 {
    let e = std::f64::consts::E;
    // sech^2(x)
    4.0 / (e.powf(-x) + e.powf(x)).sqrt()
}

// derevative of sigmoid(used in back propogation)
fn dsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

impl Network {
    pub fn display_network(&self) {
        for (i, layer) in self.layers.clone().iter().enumerate() {
            for neuron in layer.neurons.clone() {
                println!(
                    "neuron number {} with value {} \n",
                    neuron.index, neuron.value
                );
                neuron
                    .connections
                    .iter()
                    .for_each(|connection| println!("{}", connection.weight));
            }
            println!("IN LAYER {} \n \n", i);
        }
    }
    pub fn forward_propogation(&self, activation_function: &dyn Fn(f64) -> f64) {
        println!("\n \n ");
        println!("FORWARD PROPOGATION");
        for (i, layer) in self.layers.iter().enumerate() {
            if i == 0 as usize {
                continue;
            }
            for mut neuron in layer.neurons.clone() {
                neuron.set_value(activation_function(neuron.calculate()));
                println!(
                    "{value} NEW NEURON VALUE IN LAYER {ind}",
                    value = neuron.value,
                    ind = i
                );
            }
        }
        let p = self.layers.clone().last().unwrap().neurons[0].value;
        println!("{}", p);
    }

    pub fn fit_data(&mut self, data: Vec<f64>) {
        // for each data point create new
        // neuron, and push it in vector
        let mut layer = Layer {
            neurons: vec![],
            bias: rand::thread_rng().gen_range(0.1..0.9),
            index: 0,
        };
        for (i, el) in data.iter().enumerate() {
            self.number_of_neurons += i as i64;
            let neuron = Neuron {
                value: *el,
                index: self.number_of_neurons,
                layer_ind: 0,
                is_input: true,
                connections: vec![],
                activation_function: "tanh".to_string(),
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
    }
    pub fn get_output(&self) -> Vec<f64> {
        let mut output: Vec<f64> = vec![];
        println!("{}", self.layers.last().unwrap().neurons[0].value);
        for neuron in self.layers.last().unwrap().neurons.clone() {
            output.push(neuron.value);
        }
        output
    }

    pub fn create_layer(&mut self, num_of_neurons: i64) {
        self.number_of_layers += 1;
        let mut neurons = vec![];
        for i in 0..num_of_neurons {
            //  create vector of connections between last layer neurons and current neuron
            let last_layer = self.layers.last().unwrap().neurons.clone();

            self.number_of_neurons += i + 1;
            let mut neuron = Neuron {
                value: 0.0,
                layer_ind: self.number_of_layers,
                index: self.number_of_neurons,
                is_input: false,
                connections: vec![],
                activation_function: self.activation_fn.clone(),
            };

            for last_neuron in last_layer.iter() {
                let connection = Connection {
                    left_neuron: last_neuron.clone(),
                    right_neuron: Some(neuron.clone()),
                    weight: rand::thread_rng().gen_range(0.1..0.5),
                };

                neuron.connections.push(connection);
            }
            neuron.value = sigmoid(neuron.calculate());
            neurons.push(neuron);
        }

        let layer = Layer {
            neurons,
            bias: rand::thread_rng().gen_range(0.1..0.5),
            index: self.number_of_layers,
        };
        self.layers.push(layer)
    }
    fn get_neuron_by_ind(&self, index: i64) -> Option<Neuron> {
        for layer in self.layers.clone() {
            for neuron in layer.neurons {
                if neuron.index == index {
                    return Some(neuron);
                }
            }
        }
        return None;
    }
    // https://brilliant.org/wiki/backpropagation/#the-backpropagation-algorithm
    pub fn back_propogation(&self, target_values: Vec<f64>) {
        const LEARINING_RATE: f64 = 0.01;

        // TODO: 1. draw nn with more layers and calculate first weights of it.
        // 2. finish calculation of another weight in my tb
        // if final layer(we iterating over reversed layers vector)
        let last_neuron = self.layers.last().unwrap().neurons.first().unwrap().clone();
        // change in future
        let d_Err_d_Neuron = 2.0 * (last_neuron.value - target_values[0]);
        fn calculate(
            network: &Network,
            mut neuron: Neuron,
            mut intermediate: f64,
            d_Err_d_Neuron: f64,
        ) {
            if neuron.is_input {
                return;
            }
            println!("\n \n");
            println!("BACKWARD_PROPAGATION");
            println!("{} CURRENT INDEX", neuron.index);
            println!("{} LENGTH OF CURRENT CONNECTIONS", neuron.connections.len());

            // i created this variable, due to the error
            // "the cannot borrow `neuron` as immutable because it is also borrowed as mutable"
            let cloned_neuron = neuron.clone();
            for (i, connection) in neuron.connections.iter_mut().enumerate() {
                let de_dw: f64;
                // check if it's a last layer
                if neuron.layer_ind == 0 {
                    intermediate = dsigmoid(cloned_neuron.calculate()) * d_Err_d_Neuron;
                    de_dw = connection.left_neuron.value * intermediate;
                } else {
                    use math::round::ceil;
                    intermediate = intermediate
                        * dsigmoid(cloned_neuron.calculate())
                        * network
                            .get_neuron_by_ind(network.number_of_neurons) //get last neuron
                            .unwrap()
                            .connections
                            // get apropriate weight in the formula(such that it is connected with
                            // the neuron on which current differentiating weight)
                            [ceil((i / cloned_neuron.connections.len()) as f64, 0) as usize]
                            .weight;
                    de_dw = connection.left_neuron.value * intermediate;
                }
                connection.set_weight(connection.weight - LEARINING_RATE * de_dw);
                println!("{} UPDATED WEIGHT", connection.weight);
                calculate(
                    network,
                    connection.left_neuron.clone(),
                    intermediate,
                    d_Err_d_Neuron,
                );
            }
        }

        calculate(self.clone(), last_neuron, 0.0, d_Err_d_Neuron);
    }
}
