using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralOutputLayer<TOutput> : INeuralLayer<double, TOutput>
    {
        TOutput OutputValue { get; set; }
        Func<IList<double>, TOutput>  OutputValuesTransformation { get; set; }

        void Transform();
    }
}
