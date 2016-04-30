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
        private static INeuralInputLayer<double> inputLayer;
        private static INeuralNetwork<double, double> neuralNetwork; 
        private static INeuralNetworkTraining<double, double> neuralNetworkTraining;

        private static void InitializePresets()
        {
            inputLayer = new NeuralInputLayer<double>
            {
                InputValueTransformation = values => values.Select(i => i).ToList()
            };

            neuralNetwork = new NeuralNetwork<double, double>(2, 2, 2)
            {
                InputLayer = inputLayer
            };

            neuralNetworkTraining = new NeuralNetworkTraining<double, double>
            {
                NeuralNetwork = neuralNetwork,
                MaxEpochs = 100000,
                LearningRate = .5
            };
        }

        [TestMethod]
        public void TestNeuralNetTrainingData()
        {
            InitializePresets();
            neuralNetworkTraining.Train(
                new List<Tuple<IList<double>, IList<double>>>
                {
                    new Tuple<IList<double>, IList<double>>(new [] {.05, .10}, new [] {.01, .99}),
                });

            var val = new List<double> { .05, .10 };
            neuralNetwork.Evaluate(val);

            Assert.IsTrue(Math.Abs(neuralNetwork.OutputLayer.OutputValues[0] - .01) <= .005);
            Assert.IsTrue(Math.Abs(neuralNetwork.OutputLayer.OutputValues[1] - .99) <= .005);
        }
        
    }
}
