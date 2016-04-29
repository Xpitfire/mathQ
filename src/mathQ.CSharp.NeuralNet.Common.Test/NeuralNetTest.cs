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
        private static INeuralInputLayer<IList<int>> InputLayer;
        private static IList<INeuralHiddenLayer> HiddenLayers;
        private static INeuralOutputLayer<double> OutputLayer;
        private static ITrainingData<IList<int>, double> TrainingData;

        private static readonly List<int> TestData1 = new List<int> {2, -5, 4, -32, -234};
        private static readonly List<int> TestData2 = new List<int> {2, 2, 4, -32, 34};
        private static readonly List<int> TestData3 = new List<int> {2, 22, 554, -32, -234};

        private static void InitializePresets()
        {
            InputLayer = new NeuralInputLayer<IList<int>>
            {
                InputValueTransformation = values => values.Select(i => (double)i).ToList()
            };
            HiddenLayers = new List<INeuralHiddenLayer>
            {
                new NeuronHiddenLayer
                {
                    Perceptrons = new List<IPerceptron>()
                    {
                        new Perceptron
                        {
                            Weights = new List<double> { .5, .2, .3, .4, .7 },
                            Bias = 5,
                            PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction
                        },

                        new Perceptron
                        {
                            Weights = new List<double> { .1, .8, .7, .3, .1 },
                            Bias = 1,
                            PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction
                        }
                    }
                }
            };
            OutputLayer = new NeuralOutputLayer<double>
            {
                OutputValuesTransformation = values => values.Sum()
            };

            TrainingData = new TrainingData<IList<int>, double>
            {
                InputLayer = InputLayer,
                HiddenLayers = HiddenLayers,
                OutputLayer = OutputLayer
            };
        }


        [TestMethod]
        public void TestNeuralNetOneHiddenLayerTwoPerceptrons()
        {
            InitializePresets();

            var neuralNet = new NeuralNetwork<IList<int>, double>();
            neuralNet.Train(TrainingData);
            
            Assert.IsTrue(neuralNet.Evaluate(TestData1) <= .1);
            Assert.IsTrue(neuralNet.Evaluate(TestData2) >= 1.3);
            Assert.IsTrue(neuralNet.Evaluate(TestData3) >= 1.5);
        }

        [TestMethod]
        public void TestNeuralNetTrainingData()
        {
            InitializePresets();
            TrainingData = new TrainingData<IList<int>, double>(5, 2, 3)
            {
                InputLayer = InputLayer,
                OutputLayer = OutputLayer,
                MaxEpochs = 1000,
            };

            TrainingData.Randomize();
            TrainingData.Train(
                new List<IList<int>>
                {
                    new List<int> {2, -51, 34, -32, -234},
                    new List<int> {3, -3, 41, -2, -234},
                    new List<int> {-2, -77, 4, -3, -234},
                    new List<int> {-82, 5, -4, -52, 34},
                    new List<int> {23, 75, 4, -32, -234},
                    new List<int> {211, 75, 4, 42, -34},
                    new List<int> {-32, -75, -4, 432, -234},
                    new List<int> {-2, 59, 4, 32, 234},
                    new List<int> {2, 75, 4, 32, -234},
                    new List<int> {-2, 75, -4, 9, 234},
                    new List<int> {2, 95, 4, 1, 234},
                    new List<int> {2, -15, 54, 132, -234},
                    new List<int> {12, -5, -34, 32, -234}
                },
                new List<double>
                {
                    5.0,
                    1.0,
                    2.0,
                    2.5,
                    4.0,
                    1.0,
                    1.4,
                    2.0,
                    1.8,
                    4.5,
                    4.3,
                    2.5,
                    7.3
                });

            
        }

    }
}
