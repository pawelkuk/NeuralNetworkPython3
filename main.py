# -*- coding: utf-8 -*-
import mnist_loader 
import Network as net

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
network = net.Network([784,30,10])
network.SGD(training_data, 10, 10, 3.0, test_data=test_data)

