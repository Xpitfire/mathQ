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
        public double LearningRate { get; set; }
        public double Momentum { get; set; }

        public NeuralNetworkTraining()
        {
        }
        
        public void Initialize(int initialNumberOfInputValues, params int[] numberOfPerceptronsPerHiddenLayer)
        {
            if (numberOfPerceptronsPerHiddenLayer == null)
                throw new ArgumentException("Cannot initialize NeuralNetworkTraining without layer and percepton size definition!");

            var numberOfInputValues = initialNumberOfInputValues;
            NeuralNetwork.HiddenLayers = new List<INeuralHiddenLayer>(numberOfPerceptronsPerHiddenLayer.Length);
            foreach (var numberOfPerceptrons in numberOfPerceptronsPerHiddenLayer)
            {
                var perceptrons = new List<IPerceptron>();
                for (var j = 0; j < numberOfPerceptrons; j++)
                {
                    perceptrons.Add(new Perceptron
                    {
                        Weights = new double[numberOfInputValues],
                        PerceptronFunction = NeuronCalculation.ActivationFunction
                    });
                }
                NeuralNetwork.HiddenLayers.Add(new NeuronHiddenLayer
                {
                    Perceptrons = perceptrons
                });
                numberOfInputValues = numberOfPerceptrons;
            }
        }

        public void Randomize()
        {
            const double minWeight = 0.01;
            const double maxWeight = 1.0;
            const int minBias = 1;
            const int maxBias = 10;

            var random = new Random();
            const double weightRange = maxWeight - minWeight;
            const int biasRange = maxBias - minBias;
            foreach (var perceptron in NeuralNetwork.HiddenLayers.SelectMany(hiddenLayer => hiddenLayer.Perceptrons))
            {
                perceptron.Bias = random.NextDouble()*biasRange + minBias;
                for (var i = 0; i < perceptron.Weights.Count(); i++)
                {
                    perceptron.Weights[i] = random.NextDouble() * weightRange + minWeight;
                }
            }
        }
        
        public void Train(IList<Tuple<TInput, double>> trainingData)
        {
            for (var epoch = 0; epoch < MaxEpochs; epoch++)
            {
                Randomize();
                for (var idx = 0; idx < trainingData.Count; idx++)
                {
                    var prediction = ComputeHiddenLayerData(trainingData[idx].Item1);
                    for (var i = NeuralNetwork.HiddenLayers.Count - 1; i >= 0 ; i--)
                    {
                        for (var j = 0; j < NeuralNetwork.HiddenLayers[i].Perceptrons.Count; j++)
                        {
                            var perceptron = NeuralNetwork.HiddenLayers[i].Perceptrons[j];
                            var actual = trainingData[idx].Item2;
                            var z = NeuronCalculation.ZVectorFunction(
                                perceptron.InputValues, perceptron.Weights, perceptron.Bias);
                            var delta = (prediction[i][j] - actual) * 
                                (NeuronCalculation.SigmoidPrimeFunction(z));

                            perceptron.Bias = delta;

                            if (i != 0)
                            {
                                var prevPerceptrons = NeuralNetwork.HiddenLayers[i - 1].Perceptrons;
                                for (var k = 0; k < prevPerceptrons.Count; k++)
                                {
                                    perceptron.Weights[k] = delta * prevPerceptrons[k].OutputValue * perceptron.Weights[k];
                                }
                            }
                            else
                            {
                                for (var k = 0; k < perceptron.Weights.Count; k++)
                                {
                                    perceptron.Weights[k] = delta * perceptron.Weights[k];
                                }
                            }
                        }
                    }
                }
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
                        PerceptronFunction = NeuronCalculation.ActivationFunction,
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
        
        private double[][] ComputeHiddenLayerData(TInput value)
        {
            var dataSet = new double[NeuralNetwork.HiddenLayers.Count][];
            for (var i = 0; i < dataSet.Length; i++)
            {
                dataSet[i] = new double[NeuralNetwork.HiddenLayers[i].Perceptrons.Count];
            }

            NeuralNetwork.InputLayer.InitialDataSource = value;
            NeuralNetwork.InputLayer.Transform();
            var curInputValues = NeuralNetwork.InputLayer.OutputValues;
            var nextLayer = NeuralNetwork.HiddenLayers?.GetEnumerator();

            var j = 0;
            while (nextLayer?.MoveNext() ?? false)
            {
                var curLayer = nextLayer.Current;
                curLayer.InputValues = curInputValues;
                curLayer.Evaluate();

                var k = 0;
                foreach (var perceptron in curLayer.Perceptrons)
                {
                    dataSet[j][k] = perceptron.OutputValue;
                    k++;
                }
                curInputValues = curLayer.OutputValues;
                j++;
            }
            NeuralNetwork.OutputLayer.InputValues = curInputValues;
            NeuralNetwork.OutputLayer.Transform();

            return dataSet;
        }
        
    }
}
