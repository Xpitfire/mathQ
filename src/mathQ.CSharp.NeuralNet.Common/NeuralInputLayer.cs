using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralInputLayer<TInput> : INeuralInputLayer<TInput>
    {
        public IList<TInput> InputValues { get; set; }

        public IList<double> OutputValues => InputValueTransformation(InputValues);

        public Func<IList<TInput>, IList<double>> InputValueTransformation { get; set; }
    }
}
