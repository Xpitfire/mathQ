using System;
using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralInputLayer<TInput> : INeuralLayer<TInput, double>
    {
        Func<IList<TInput>, IList<double>> InputValueTransformation { get; set; }
        void Transform();
    }
}