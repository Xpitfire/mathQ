using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralOutputLayer<TOutput> : INeuralOutputLayer<TOutput>
    {
        public IList<double> InputValues { get; set; }
        public IList<double> OutputValues { get; set; }
        public IList<IPerceptron> Perceptrons { get; set; }
        public TOutput OutputValue => OutputValuesTransformation != null ? OutputValuesTransformation(OutputValues) : default(TOutput);

        public Func<IList<double>, TOutput> OutputValuesTransformation { get; set; }

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
