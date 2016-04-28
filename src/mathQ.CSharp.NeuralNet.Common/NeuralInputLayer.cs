﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mathQ.CSharp.NeuralNet.Common
{
    public class NeuralInputLayer<TInput> : INeuralInputLayer<TInput>
    {
        public IEnumerable<TInput> InputValues { get; set; }
        public IEnumerable<double> OutputValues { get; set; }
        public Func<TInput, double> InputValueTransformation { get; set; }
        public void Transform()
        {
            OutputValues = InputValues.Select(
                inputValue => InputValueTransformation(inputValue)).ToList();
        }
    }
}
