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
            NeuralNetwork.OutputLayer = new NeuralOutputLayer<TOutput>
            {
                Perceptrons = new List<IPerceptron>(numberOfInputValues)
            };
            for (var i = 0; i < numberOfInputValues; i++)
            {
                NeuralNetwork.OutputLayer.Perceptrons.Add(new Perceptron
                {
                    Weights = new double[numberOfInputValues],
                    PerceptronFunction = NeuronCalculation.ActivationFunction
                });
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
        
        public void Train(IList<Tuple<IList<TInput>, IList<double>>> trainingData)
        {
            Randomize();
            for (var epoch = 0; epoch < MaxEpochs; epoch++)
            {
                foreach (var data in trainingData)
                {
                    NeuralNetwork.Evaluate(data.Item1);
                    IList<double> computedValues = NeuralNetwork.OutputLayer.OutputValues;
                    IList<double> actualValues = data.Item2;

                    IList<INeuralHiddenLayer> layers = NeuralNetwork.HiddenLayers;
                    INeuralOutputLayer<TOutput> outputLayer = NeuralNetwork.OutputLayer;

                    IList<IPerceptron> perceptrons = outputLayer.Perceptrons;
                    
                    for (var i = layers.Count - 1; layers.Count > 0; i--)
                    {
                        var curLayer = layers[i];
                        IList<double> delta = NeuronCalculation.VectorSubtraction(actualValues, computedValues);
                        double prevLayerDelta = NeuronCalculation.VectorDotProduct(delta, NeuronCalculation.GradientSigmoidFunction(computedValues));
                        UpdateWeights(perceptrons, prevLayerDelta, layers[i]);
                        actualValues = layers[i].OutputValues;
                        computedValues = layers[i - 1].OutputValues;
                        perceptrons = layers[i - 1].Perceptrons;
                    }
                    
                }
            }
        }

        private void UpdateWeights(IList<IPerceptron> perceptrons, double prevLayerDelta, INeuralHiddenLayer neuralHiddenLayer)
        {
            foreach (var perceptron in perceptrons)
            {
                perceptron.Weights = NeuronCalculation.VectorAdd(perceptron.Weights, 
                    NeuronCalculation.VectorScalarMultiplication(prevLayerDelta, neuralHiddenLayer.OutputValues));
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
        
        private double[][] ComputeHiddenLayerData(IList<double> values)
        {
            var numberOfLayers = NeuralNetwork.HiddenLayers.Count + 1;
            var dataSet = new double[numberOfLayers][];
            for (var i = 0; i < dataSet.Length; i++)
            {
                dataSet[i] = new double[NeuralNetwork.HiddenLayers[i].Perceptrons.Count];
            }
            
            var curInputValues = values;
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
            dataSet[numberOfLayers - 1] = NeuralNetwork.OutputLayer.OutputValues.ToArray(); 

            return dataSet;
        }
        
    }
}
