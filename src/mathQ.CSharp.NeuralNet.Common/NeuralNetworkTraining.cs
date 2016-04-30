using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mathQ.CSharp.NeuralNet.Function;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralNetworkTraining<TInput, TOutput> : INeuralNetworkTraining<TInput, TOutput>
    {
        public INeuralNetwork<TInput, TOutput> NeuralNetwork { get; set; }
        public int MaxEpochs { get; set; }
        public double LearningRate { get; set; } = 1.0;

        public void Randomize()
        {
            const double minWeight = 0.01;
            const double maxWeight = 1.0;
            const double minBias = .1;
            const double maxBias = 1.0;

            var random = new Random();
            const double weightRange = maxWeight - minWeight;
            const double biasRange = maxBias - minBias;
            foreach (var perceptron in NeuralNetwork.HiddenLayers.SelectMany(hiddenLayer => hiddenLayer.Perceptrons))
            {
                perceptron.Bias = random.NextDouble()*biasRange + minBias;
                for (var i = 0; i < perceptron.Weights.Count(); i++)
                {
                    perceptron.Weights[i] = random.NextDouble() * weightRange + minWeight;
                }
            }
            foreach (var perceptron in NeuralNetwork.OutputLayer.Perceptrons)
            {
                perceptron.Bias = random.NextDouble() * biasRange + minBias;
                for (var i = 0; i < perceptron.Weights.Count(); i++)
                {
                    perceptron.Weights[i] = random.NextDouble() * weightRange + minWeight;
                }
            }
        }
        
        public void Train(IList<Tuple<IList<TInput>, IList<double>>> trainingData)
        {
            Randomize();
            for (var epoch = 0; epoch < MaxEpochs; epoch++)
            {
                foreach (var data in trainingData)
                {
                    NeuralNetwork.Evaluate(data.Item1);
                    var deltaGradientList = new List<double>();
                    var prevOutValues = new List<double>();
                    var prevWeights = new List<double>();
                    var perceptrons = NeuralNetwork.OutputLayer.Perceptrons;
                    for (var i = 0; i < perceptrons.Count; i++)
                    {
                        var targetOut = data.Item2[i];
                        var actualOut = NeuralNetwork.OutputLayer.OutputValues[i];
                        var deltaOut = actualOut - targetOut;
                        var gradientActualOut = NeuronCalculation.GradientFunction(actualOut);
                        var deltaGradient = deltaOut*gradientActualOut;
                        var prevOutValue = perceptrons[i].InputValues[i];
                        deltaGradientList.Add(deltaGradient);
                        prevOutValues.Add(prevOutValue);
                        var outChangeRate = deltaGradient*prevOutValue;
                        for (var j = 0; j < perceptrons[i].Weights.Count; j++)
                        {
                            prevWeights.Add(perceptrons[i].Weights[j]);
                            perceptrons[i].Weights[j] = perceptrons[i].Weights[j] - LearningRate*outChangeRate;
                        }
                    }

                    foreach (var hiddenLayer in NeuralNetwork.HiddenLayers)
                    {
                        var inputValues = hiddenLayer.InputValues;
                        for (var i = 0; i < hiddenLayer.Perceptrons.Count; i++)
                        {
                            var perceptron = hiddenLayer.Perceptrons[i];
                            var errorSum = 0.0;
                            for (var j = 0; j < deltaGradientList.Count; j++)
                            {
                                errorSum += deltaGradientList[j]*prevWeights[i + j*hiddenLayer.Perceptrons.Count];
                            }

                            for (var j = 0; j < perceptron.Weights.Count; j++)
                            {
                                var gradientOut0 = NeuronCalculation.GradientFunction(perceptron.OutputValue);
                                var inChangeRate0 = errorSum*gradientOut0*inputValues[j];
                                perceptron.Weights[j] = perceptron.Weights[j] - LearningRate*inChangeRate0;
                            }
                        }
                    }
                }
            }
            
        }

        private void UpdateWeights(IList<IPerceptron> perceptrons, double prevLayerDelta, INeuralHiddenLayer neuralHiddenLayer)
        {
            foreach (var perceptron in perceptrons)
            {
                perceptron.Weights = NeuronCalculation.VectorScalarMultiplication(prevLayerDelta, neuralHiddenLayer.OutputValues);
            }
        }

        public void InitializePresets(double[][][] values)
        {
            NeuralNetwork.HiddenLayers = new INeuralHiddenLayer[values.Length];
            foreach (var perceptronVal in values)
            {
                var hiddenLayer = new NeuronHiddenLayer
                {
                    Perceptrons = new IPerceptron[perceptronVal.Length]
                };
                foreach (var weightVal in perceptronVal)
                {
                    var perceptron = new Perceptron
                    {
                        Weights = new double[weightVal.Length]
                    };
                    var len = weightVal.Length - 1;
                    for (var k = 0; k < len; k++)
                    {
                        perceptron.Weights[k] = weightVal[k];
                    }
                    perceptron.Bias = weightVal[len];
                    hiddenLayer.Perceptrons.Add(perceptron);
                }
                NeuralNetwork.HiddenLayers.Add(hiddenLayer);
            }
        }
    }
}
