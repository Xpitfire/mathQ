using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralNetwork<TInput, TOutput> : INeuralNetwork<TInput, TOutput>
    {
        public INeuralInputLayer<TInput> InputLayer { get; set; }
        public IList<INeuralHiddenLayer> HiddenLayers { get; set; }
        public INeuralOutputLayer<TOutput> OutputLayer { get; set; }

        public NeuralNetwork(int initialNumberOfInputValues, int initialNumberOfOutputValues, params int[] numberOfPerceptronsPerHiddenLayer)
        {
            Initialize(initialNumberOfInputValues, initialNumberOfOutputValues, numberOfPerceptronsPerHiddenLayer);
        }

        private void Initialize(int initialNumberOfInputValues, int initialNumberOfOutputValues, params int[] numberOfPerceptronsPerHiddenLayer)
        {
            if (numberOfPerceptronsPerHiddenLayer == null)
                throw new ArgumentException("Cannot initialize NeuralNetworkTraining without layer and percepton size definition!");

            var numberOfInputValues = initialNumberOfInputValues;
            HiddenLayers = new List<INeuralHiddenLayer>(numberOfPerceptronsPerHiddenLayer.Length);
            foreach (var numberOfPerceptrons in numberOfPerceptronsPerHiddenLayer)
            {
                var perceptrons = new List<IPerceptron>();
                for (var j = 0; j < numberOfPerceptrons; j++)
                {
                    perceptrons.Add(new Perceptron
                    {
                        Weights = new double[numberOfInputValues]
                    });
                }
                HiddenLayers.Add(new NeuronHiddenLayer
                {
                    Perceptrons = perceptrons
                });
                numberOfInputValues = numberOfPerceptrons;
            }
            OutputLayer = new NeuralOutputLayer<TOutput>
            {
                Perceptrons = new List<IPerceptron>(numberOfInputValues)
            };
            for (var i = 0; i < initialNumberOfOutputValues; i++)
            {
                OutputLayer.Perceptrons.Add(new Perceptron
                {
                    Weights = new double[numberOfInputValues]
                });
            }
        }

        public TOutput Evaluate(IList<TInput> value)
        {
            InputLayer.InputValues = value;
            var curInputValues = InputLayer.OutputValues;
            var nextLayer = HiddenLayers?.GetEnumerator();
            while (nextLayer?.MoveNext() ?? false)
            {
                var curLayer = nextLayer.Current;
                curLayer.InputValues = curInputValues;
                curLayer.Evaluate();
                curInputValues = curLayer.OutputValues;
            }
            OutputLayer.InputValues = curInputValues;
            OutputLayer.Evaluate();
            return OutputLayer.OutputValue;
        }
        
    }
}
