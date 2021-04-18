#[derive(Clone)]
struct Neuron{
    value: f64,
    bias: f64,
    connections: Vec<Connection>,
    activation_function: String,
}

impl Neuron{
    fn calculate(&mut self) -> f64{

        // neuron = some activation function(sigmoid, tanh and etc.)(sum of all weights multiplyed by all inputs + bias)
        let mut sum: f64 = 0.0;
        for connection in self.connections.iter(){
            sum+=connection.left_neuron.value as f64 * connection.weight as f64;
        }
        sum+=self.bias as f64;
        sum.tanh()
    }
}

#[derive(Clone)]
struct Connection{
    left_neuron: Neuron,
    weight: f64,
}

#[derive(Clone)]
struct Layer{
    neurons: Vec<Neuron> 
}

struct Network{
    layers: Vec<Layer>
}

impl Network{
    fn fit_data(&mut self, data: Vec<f64>){
        // for each data point create new 
        // neuron, and push it in vector
        for i in data.iter(){
            let neuron = Neuron{
                value: *i,
                bias: 0.0,
                connections: vec![], activation_function: "tanh".to_string(),
            };
            self.layers[0].neurons.push(neuron);
        }
    }

    fn create_layer(&mut self, num_of_neurons: i64){
        let mut neurons = vec![];
        for _ in 0..num_of_neurons{

            //  create vector of connections between last layer neurons and current neuron
            let mut connections = vec![]; 
            let last_layer = self.layers.last().unwrap().neurons.clone();
            for neuron in last_layer.iter(){
                let connection = Connection{
                    left_neuron: neuron.clone(),
                    weight: 0.5,
                };
                connections.push(connection);
            }

            let mut neuron = Neuron{
                value: 0.0,
                bias: 0.0,
                connections,
                activation_function: "tanh".to_string()
            };

            neuron.calculate();
            neurons.push(neuron);
        }

        let layer = Layer{
            neurons
        };
        self.layers.push(layer)
    }
    
    //TODO: https://www.edureka.co/blog/backpropagation/
    // https://brilliant.org/wiki/backpropagation/#the-backpropagation-algorithm
    // n - learning rate(0.001 for example)
    fn back_propogation(){

    }
}

