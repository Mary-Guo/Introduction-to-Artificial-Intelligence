import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        percentage = 0
        while percentage < 1:
            correct = 0
            size = 0
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) == nn.as_scalar(y):
                    correct += 1
                else:
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                size += 1
            percentage = correct/size 



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Parameter(1, 300)
        self.m2 = nn.Parameter(300, 1)
        self.b1 = nn.Parameter(1, 300)
        self.b2 = nn.Parameter(1, 1)
        self.learning_rate = 0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x, self.m1)
        layer_one = nn.ReLU(nn.AddBias(xm, self.b1))
        layer_two = nn.AddBias(nn.Linear(layer_one, self.m2), self.b2)
        return layer_two


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        boolean = True
        while boolean:
            for x, y in dataset.iterate_once(200):
                loss = self.get_loss(x,y)

                if nn.as_scalar(loss) < 0.02:
                    return 
                m1 = self.m1
                m2 = self.m2
                b1 = self.b1
                b2 = self.b2
                grad_wrt_m1, grad_wrt_m2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [m1, m2, b1, b2])
                self.m1.update(grad_wrt_m1, -self.learning_rate)
                self.m2.update(grad_wrt_m2, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Parameter(784, 1200)
        self.m2 = nn.Parameter(1200, 10)
        #self.m3 = nn.Parameter(128, 10)
        self.b1 = nn.Parameter(1, 1200)
        self.b2 = nn.Parameter(1, 10)
        #self.b3 = nn.Parameter(1, 10)

        self.learning_rate = 0.5


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x, self.m1)
        layer_one = nn.ReLU(nn.AddBias(xm, self.b1))

        layer_two = nn.AddBias(nn.Linear(layer_one, self.m2), self.b2)



        return layer_two

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        boolean = True
        while boolean:
            for x, y in dataset.iterate_once(200):
                loss = self.get_loss(x,y)
                accuracy = dataset.get_validation_accuracy()
                if accuracy >= 0.975:
                    return 
                m1 = self.m1
                m2 = self.m2
                #m3 = self.m3
                b1 = self.b1
                b2 = self.b2
                #b3 = self.b3
                grad_wrt_m1, grad_wrt_m2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [m1, m2, b1, b2])
                self.m1.update(grad_wrt_m1, -self.learning_rate)
                self.m2.update(grad_wrt_m2, -self.learning_rate)
                #self.m3.update(grad_wrt_m3, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
                #self.b3.update(grad_wrt_b3, -self.learning_rate)

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        self.w1 = nn.Parameter(self.state_size, 300)
        self.b1 = nn.Parameter(1, 300)
        self.w2 = nn.Parameter(300, 300)
        self.b2 = nn.Parameter(1, 300)
        self.w3 = nn.Parameter(300, self.num_actions)
        self.b3 = nn.Parameter(1, self.num_actions)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]



        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.8
        self.numTrainingGames = 2100
        self.batch_size = 64

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        predicted_q = self.run(states)
        return nn.SquareLoss(predicted_q, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

        

        mx = nn.Linear(states, self.w1)
        layer_one = nn.ReLU(nn.AddBias(mx, self.b1))
        layer_two = nn.ReLU(nn.AddBias(nn.Linear(layer_one, self.w2), self.b2))
        layer_three = nn.AddBias(nn.Linear(layer_two, self.w3), self.b3)   

        return layer_three   

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        grad = nn.gradients(self.get_loss(states, Q_target), self.parameters)
        for i in range(len(self.parameters)):
            self.parameters[i].update(grad[i], -self.learning_rate)

