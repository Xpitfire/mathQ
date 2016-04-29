using System;
using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralInputLayer<TInput> : INeuralLayer<TInput, double>
    {
        TInput InitialDataSource { get; set; }

        Func<TInput, IList<double>> InputValueTransformation { get; set; }
        void Transform();
    }
}