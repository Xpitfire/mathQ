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
        public IEnumerable<INeuron<double, double>> Perceptrons { get; set; }
        
        public void Evaluate()
        {
            var values = new List<double>();
            foreach (var perceptron in Perceptrons)
            {
                perceptron.InputValues = InputValues;
                perceptron.Evaluate();
                values.Add(perceptron.OutputValue);
            }
            OutputValues = values;
        }
    }
}
