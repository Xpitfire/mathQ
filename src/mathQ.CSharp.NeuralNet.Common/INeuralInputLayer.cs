using System;
using System.Collections.Generic;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralInputLayer<TInput> : INeuralLayer<TInput, double>
    {
        Func<TInput, double> InputValueTransformation { get; set; }
        void Transform();
    }
}