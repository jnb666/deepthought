package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"math"
)

// Layer interface type represents one layer in the network.
type Layer interface {
	FeedForward(in blas.Matrix) blas.Matrix
	BackProp(err blas.Matrix, eta float64) blas.Matrix
	Weights() blas.Matrix
	Gradient() blas.Matrix
	Cost(t blas.Matrix) float64
	Release()
}

type layer struct {
	nlayer   int
	activ    Activation
	input    blas.Matrix // Z matrix of value at each node [samples, nin+1]
	inputT   blas.Matrix // Z matrix transposed [nin+1, samples]
	output   blas.Matrix // return value at each node [samples, nout]
	weights  blas.Matrix // W outgoing weight matrix  [nout, nin+1]
	weightT  blas.Matrix // W matrix transpose without biases  [nin, nout]
	gradient blas.Matrix // G gradient of weight matrix [nout, nin+1]
	deriv    blas.Matrix // Fp matrix of derivative of activation fn [samples, nin]
	delta    blas.Matrix // D matrix of errors at each node [samples, nin]
	deltaT   blas.Matrix // D matrix transposed [nin, samples]
}

// AddLayer method adds a new input or hidden layer to the network.
func (n *Network) AddLayer(nin, nout int, a Activation) {
	batch := n.BatchSize
	l := &layer{
		nlayer:   n.Layers,
		activ:    a,
		input:    blas.New(batch, nin+1),
		inputT:   blas.New(nin+1, batch),
		output:   blas.New(batch, nout),
		weights:  blas.New(nout, nin+1),
		gradient: blas.New(nout, nin+1),
		weightT:  blas.New(nin, nout),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nin)
	}
	if n.Layers > 0 {
		l.delta = blas.New(nin, batch)
		l.deltaT = blas.New(batch, nin)
	}
	n.Add(l)
}

func (l *layer) Release() {
	l.input.Release()
	l.inputT.Release()
	l.output.Release()
	l.weights.Release()
	l.gradient.Release()
	l.weightT.Release()
	if l.deriv != nil {
		l.deriv.Release()
	}
	if l.delta != nil {
		l.delta.Release()
		l.deltaT.Release()
	}
}

func (l *layer) Weights() blas.Matrix { return l.weights }

func (l *layer) Gradient() blas.Matrix { return l.gradient }

func (l *layer) Cost(t blas.Matrix) float64 { panic("no cost for input or hidden layer!") }

func (l *layer) FeedForward(in blas.Matrix) blas.Matrix {
	l.activ.Func.Apply(in, l.input)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(l.input, l.deriv)
	}
	nin := in.Cols()
	l.input.Reshape(in.Rows(), nin+1, false)
	l.input.Col(nin, nin+1).Set(1)
	if Debug {
		fmt.Printf("Layer %d feedforward: input\n%s\n", l.nlayer, l.input)
	}
	l.output.Mul(l.input, l.weights)
	if Debug {
		fmt.Printf("Layer %d feedforward: output\n%s\n", l.nlayer, l.output)
	}
	return l.output
}

func (l *layer) BackProp(err blas.Matrix, eta float64) blas.Matrix {
	if Debug {
		fmt.Printf("Layer %d backprop: input delta\n%s\n", l.nlayer, err)
	}
	// update gradient
	etap := -eta / float64(l.output.Rows())
	l.inputT.Transpose(l.input)
	l.output.Transpose(err)
	l.gradient.Mul(l.output, l.inputT) // [nout, samples] x [samples, nin+1]
	l.gradient.Scale(etap)             // [nout, nin+1]
	if Debug {
		fmt.Printf("Layer %d backprop: gradient\n%s\n", l.nlayer, l.gradient)
	}
	// propagate error backward
	if l.delta != nil {
		if Debug {
			fmt.Printf("Layer %d backprop: weights\n%s\n", l.nlayer, l.weights)
		}
		c := l.weights.Cols()
		l.weightT.Transpose(l.weights.Col(0, c-1)) // [nin, nout]
		l.deltaT.Mul(l.weightT, err)               // [nin, nout] x [nout, samples]
		l.delta.Transpose(l.deltaT)                // [samples, nin]
		l.delta.MulElem(l.delta, l.deriv)
		if Debug {
			fmt.Printf("Layer %d backprop: output delta\n%s\n", l.nlayer, l.delta)
		}
	}
	return l.delta
}

type outLayer struct {
	activ  Activation
	values blas.Matrix // Z matrix of values at each node [samples, nodes]
	delta  blas.Matrix // D matrix of errors at each node [samples, nodes]
	deriv  blas.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	temp   blas.Matrix
	cost   blas.BinaryFunction
}

func newOutLayer(batch, nodes int, a Activation) *outLayer {
	l := &outLayer{
		activ:  a,
		values: blas.New(batch, nodes),
		delta:  blas.New(nodes, batch),
		temp:   blas.New(batch, nodes),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nodes)
	}
	return l
}

func (l *outLayer) Release() {
	l.values.Release()
	l.delta.Release()
	l.temp.Release()
	if l.deriv != nil {
		l.deriv.Release()
	}
}

func (l *outLayer) Weights() blas.Matrix { panic("no weights for output layer!") }

func (l *outLayer) Gradient() blas.Matrix { panic("no gradient for output layer!") }

func (l *outLayer) FeedForward(in blas.Matrix) blas.Matrix {
	if Debug {
		fmt.Printf("OutLayer feedforward: input\n%s\n", in)
	}
	l.activ.Func.Apply(in, l.values)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(l.values, l.deriv)
	}
	if Debug {
		fmt.Printf("OutLayer feedforward: output\n%s\n", l.values)
	}
	return l.values
}

func (l *outLayer) BackProp(target blas.Matrix, eta float64) blas.Matrix {
	l.delta.Add(l.values, target, -1)
	if l.activ.Deriv != nil {
		l.delta.MulElem(l.delta, l.deriv)
	}
	if Debug {
		fmt.Printf("OutLayer backprop: delta\n%s\n", l.delta)
	}
	return l.delta
}

func (l *outLayer) Cost(target blas.Matrix) float64 {
	l.cost.Apply(l.values, target, l.temp)
	return l.temp.Sum() / float64(target.Rows())
}

// QuadraticOutput method appends a quadratic cost output layer to the network.
func (n *Network) QuadraticOutput(nodes int, a Activation) {
	layer := newOutLayer(n.BatchSize, nodes, a)
	if blas.Implementation() == blas.OpenCL32 {
		layer.cost = blas.NewBinaryCL("(x - y) * (x - y)")
	} else {
		layer.cost = blas.Binary64(func(out, tgt float64) float64 { return (out - tgt) * (out - tgt) })
	}
	n.Add(layer)
}

// CrossEntropyOutput method appends a cross entropy output layer with softmax activation to the network.
func (n *Network) CrossEntropyOutput(nodes int) {
	layer := newOutLayer(n.BatchSize, nodes, Softmax)
	if blas.Implementation() == blas.OpenCL32 {
		layer.cost = blas.NewBinaryCL("-y * log(max(x, 1e-10f))")
	} else {
		layer.cost = blas.Binary64(func(out, tgt float64) float64 {
			if out < 1e-10 {
				out = 1e-10
			}
			return -tgt * math.Log(out)
		})
	}
	n.Add(layer)
}
