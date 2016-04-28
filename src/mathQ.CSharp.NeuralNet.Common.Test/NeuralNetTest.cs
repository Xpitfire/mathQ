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
        public void TestNeuralNetOneHiddenLayerTwoPerceptrons()
        {
            var inputLayer = new NeuralInputLayer<int>
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
                            Weights = new List<double> { .5, .2, .3, .4, .7 },
                            Biases = new List<double> { 1, 2, 3, 4, 5 },
                            PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction
                        },

                        new Perceptron
                        {
                            Weights = new List<double> { .1, .8, .7, .3, .1 },
                            Biases = new List<double> { 5, 4, 3, 2, 1 },
                            PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction
                        }
                    }
                }
            };
            var outputLayer = new NeuralOutputLayer<double>
            {
                OutputValuesTransformation = values => values.Sum()
            };

            var trainingData = new TrainingData<int, double>
            {
                InputLayer = inputLayer,
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer
            };

            var neuralNet = new NeuralNetwork<int, double>();
            neuralNet.Train(trainingData);
            
            Assert.IsTrue(neuralNet.Evaluate(new List<int> { 2, -5, 4, -32, -234 }) < .1);
            Assert.IsTrue(neuralNet.Evaluate(new List<int> {2, 2, 4, -32, 34}) > 1.5);
            Assert.IsTrue(neuralNet.Evaluate(new List<int> {2, 22, 554, -32, -234}) > 1.9);
        }
    }
}
