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
        private static INeuralInputLayer<IList<int>> inputLayer;
        private static IList<INeuralHiddenLayer> hiddenLayers;
        private static INeuralOutputLayer<double> outputLayer;
        private static INeuralNetwork<IList<int>, double> neuralNetwork; 
        private static INeuralNetworkTraining<IList<int>, double> neuralNetworkTraining;

        private static readonly List<int> TestData1 = new List<int> {2, -5, 4, -32, -234};
        private static readonly List<int> TestData2 = new List<int> {2, 2, 4, -32, 34};
        private static readonly List<int> TestData3 = new List<int> {2, 22, 554, -32, -234};

        private static void InitializePresets(int initialNumberOfInputValues, params int[] numberOfPerceptronsPerHiddenLayer)
        {
            inputLayer = new NeuralInputLayer<IList<int>>
            {
                InputValueTransformation = values => values.Select(i => (double)i).ToList()
            };

            hiddenLayers = new List<INeuralHiddenLayer>
            {
                new NeuronHiddenLayer
                {
                    Perceptrons = new List<IPerceptron>()
                    {
                        new Perceptron
                        {
                            Weights = new List<double> { .5, .2, .3, .4, .7 },
                            Bias = 5,
                            PerceptronFunction = NeuronCalculation.ActivationFunction
                        },

                        new Perceptron
                        {
                            Weights = new List<double> { .1, .8, .7, .3, .1 },
                            Bias = 1,
                            PerceptronFunction = NeuronCalculation.ActivationFunction
                        }
                    }
                }
            };

            outputLayer = new NeuralOutputLayer<double>
            {
                OutputValuesTransformation = values => values.Sum()
            };

            neuralNetwork = new NeuralNetwork<IList<int>, double>
            {
                InputLayer = inputLayer,
                HiddenLayers = hiddenLayers,
                OutputLayer = outputLayer
            };

            neuralNetworkTraining = new NeuralNetworkTraining<IList<int>, double>
            {
                NeuralNetwork = neuralNetwork,
                MaxEpochs = 1000,
            };
            neuralNetworkTraining.Initialize(initialNumberOfInputValues, numberOfPerceptronsPerHiddenLayer);
        }


        [TestMethod]
        public void TestNeuralNetOneHiddenLayerTwoPerceptrons()
        {
            InitializePresets(5, 2);
            
            Assert.IsTrue(neuralNetwork.Evaluate(TestData1) <= .1);
            Assert.IsTrue(neuralNetwork.Evaluate(TestData2) >= 1.3);
            Assert.IsTrue(neuralNetwork.Evaluate(TestData3) >= 1.5);
        }

        [TestMethod]
        public void TestNeuralNetTrainingData()
        {
            InitializePresets(5, 7, 5);

            neuralNetworkTraining.Train(
                new List<Tuple<IList<int>, double>>
                {
                    new Tuple<IList<int>, double>(new List<int> {1, 0, 0, 0, 0}, 1.0),
                    new Tuple<IList<int>, double>(new List<int> {1, 0, 0, 0, 0}, 1.0),
                    new Tuple<IList<int>, double>(new List<int> {1, 0, 0, 0, 0}, 1.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 1, 0, 0, 0}, 2.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 1, 0, 0, 0}, 2.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 1, 0, 0, 0}, 2.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 1, 0, 0, 0}, 2.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 1, 0, 0}, 3.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 1, 0, 0}, 3.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 1, 0, 0}, 3.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 0, 1, 0}, 4.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 0, 1, 0}, 4.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 0, 0, 1}, 5.0),
                    new Tuple<IList<int>, double>(new List<int> {0, 0, 0, 0, 1}, 5.0)
                });
            var val = new List<int> {1, 0, 0, 0, 0};
            var res = neuralNetwork.Evaluate(val);
            var val2 = new List<int> {0, 0, 0, 0, 1};
            var res2 = neuralNetwork.Evaluate(val2);

            Assert.IsTrue(Math.Abs(res - 1.0) <= 0.01);
        }

    }
}
