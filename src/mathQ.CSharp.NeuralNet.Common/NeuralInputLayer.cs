using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralInputLayer<TInput> : INeuralInputLayer<TInput>
    {
        public TInput InitialDataSource { get; set; }
        public IList<TInput> InputValues { get; set; }
        public IList<double> OutputValues { get; set; }
        public Func<TInput, IList<double>> InputValueTransformation { get; set; }
        public void Transform()
        {
            OutputValues = InputValueTransformation(InitialDataSource);
        }
    }
}
