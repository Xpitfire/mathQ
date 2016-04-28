using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralOutputLayer<TOutput> : INeuralOutputLayer<TOutput>
    {
        public IEnumerable<double> InputValues { get; set; }
        public IEnumerable<TOutput> OutputValues { get; set; }
        public TOutput OutputValue { get; set; }
        public Func<IEnumerable<double>, TOutput> OutputValuesTransformation { get; set; }

        public void Transform()
        {
            OutputValue = OutputValuesTransformation(InputValues);
            OutputValues = new[] {OutputValue};
        }
    }
}
