using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuronHiddenLayer : INeuralHiddenLayer
    {
        public IEnumerable<double> InputValues { get; set; }
        public IEnumerable<double> OutputValues { get; set; }
        public event NeuronSignal NeuronSignal;
        public IEnumerable<IPerceptron> Perceptrons { get; }
        public IEnumerable<double> Evaluate(IEnumerable<double> inputValues)
        {
            throw new NotImplementedException();
        }

        public void TrainPerceptrons(IReadOnlyList<double> weights, IReadOnlyList<double> biases)
        {
            foreach (var perceptron in Perceptrons)
            {
                perceptron.Weights = weights;
                perceptron.Biases = biases;
            }
        }
    }
}
