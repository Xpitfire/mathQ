using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mathQ.CSharp.NeuralNet.Function;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class TrainingData<TInput, TOutput> : ITrainingData<TInput, TOutput>
    {
        public INeuralInputLayer<TInput> InputLayer { get; set; }
        public IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        public INeuralOutputLayer<TOutput> OutputLayer { get; set; }

        public int MaxEpochs { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }

        public TrainingData()
        {
        }

        public TrainingData(int initialNumberOfInputValues, params int[] numberOfPerceptronsPerHiddenLayer)
        {
            if (numberOfPerceptronsPerHiddenLayer == null)
                throw new ArgumentException("Cannot initialize TrainingData without layer and percepton size definition!");

            var numberOfInputValues = initialNumberOfInputValues;
            HiddenLayers = new List<INeuralHiddenLayer>(numberOfPerceptronsPerHiddenLayer.Length);
            foreach (var numberOfPerceptrons in numberOfPerceptronsPerHiddenLayer)
            {
                var perceptrons = new List<IPerceptron>();
                for (var j = 0; j < numberOfPerceptrons; j++)
                {
                    perceptrons.Add(new Perceptron
                    {
                        Weights = new double[numberOfInputValues],
                        PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction
                    });
                }
                HiddenLayers.Add(new NeuronHiddenLayer
                {
                    Perceptrons = perceptrons
                });
                numberOfInputValues = numberOfPerceptrons;
            }
        }

        public void Randomize(double minWeight, double maxWeight, double minBias, double maxBias)
        {
            var random = new Random();
            var weightRange = maxWeight - minWeight;
            var biasRange = maxBias - minBias;
            foreach (var perceptron in HiddenLayers.SelectMany(hiddenLayer => hiddenLayer.Perceptrons))
            {
                perceptron.Bias = random.NextDouble()*biasRange + minBias;
                for (var i = 0; i < perceptron.Weights.Count(); i++)
                {
                    perceptron.Weights[i] = random.NextDouble() * weightRange + minWeight;
                }
            }
        }

        public void Train(IList<TInput> trainingValues)
        {
            for (var e = 0; e < MaxEpochs; e++)
            {
                foreach (var dataSet in trainingValues.Select(ComputeHiddenLayerData))
                {
                    for (var i = 0; i < HiddenLayers.Count; i++)
                    {
                        for (var j = 0; j < HiddenLayers[i].Perceptrons.Count; j++)
                        {
                            var output = HiddenLayers[i].Perceptrons[j].OutputValue;
                            var error = output*(1 - output)*(dataSet[i][j] - output);

                            for (var k = 0; k < HiddenLayers[i].Perceptrons[j].Weights.Count; k++)
                            {
                                HiddenLayers[i].Perceptrons[j].Weights[k] 
                                    += error*HiddenLayers[i].Perceptrons[j].InputValues[k];
                            }
                            HiddenLayers[i].Perceptrons[j].Bias += error;
                        }
                    }
                }
            }
        }

        public void InitializePresets(double[][][] values)
        {
            HiddenLayers = new INeuralHiddenLayer[values.Length];
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
                        PerceptronFunction = NeuronCalculation.PerceptronSigmoidFunction,
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
                HiddenLayers.Add(hiddenLayer);
            }
        }

        public double[][] ComputeHiddenLayerData(TInput value)
        {
            var dataSet = new double[HiddenLayers.Count][];
            for (var i = 0; i < dataSet.Length; i++)
            {
                dataSet[i] = new double[HiddenLayers[i].Perceptrons.Count];
            }

            InputLayer.InitialDataSource = value;
            InputLayer.Transform();
            var curInputValues = InputLayer.OutputValues;
            var nextLayer = HiddenLayers?.GetEnumerator();

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
            OutputLayer.InputValues = curInputValues;
            OutputLayer.Transform();

            return dataSet;
        }
        
    }
}
