﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public interface INeuralHiddenLayer : INeuralLayer<double, double>
    {
        IList<IPerceptron> Perceptrons { get; set; }

        void Evaluate();
    }
}
