using System;
using System.Collections.Generic;
using System.Linq;
using mathQ.CSharp.NeuralNet.Function;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace mathQ.CSharp.NeuralNet.Common.Test
{
    [TestClass]
    public class NeuralNetTest
    {
        [TestMethod]
        public void TestNeuralNetOneHiddenLayerOnePerceptrons()
        {
            var inputLayer = new NeuralInputLayer<char>
            {
                InputValueTransformation = chr => chr
            };
            var hiddenLayers = new List<INeuralHiddenLayer>
            {
                new NeuronHiddenLayer
                {
                    Perceptrons = new List<IPerceptron>()
                    {
                        new Perceptron
                        {
                            Weights = new List<double> { 2.5, 6.4 },
                            Biases = new List<double> { 5, 3 },
                            ValueTransformation = NeuronCalculation.PerceptronFunction
                        }
                    }
                }
            };
            var outputLayer = new NeuralOutputLayer<int>
            {
                OutputValuesTransformation = values => (int) values.Sum()
            };

            var trainingData = new TrainingData<char, int>
            {
                InputLayer = inputLayer,
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer
            };

            var neuralNet = new NeuralNetwork<char, int>();
            neuralNet.Train(trainingData);

            var result = neuralNet.Evaluate(new List<char> { 'A', 'C' });
            Console.WriteLine($"NeuralNet-Output: {result}");
        }
    }
}
