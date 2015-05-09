package network

import (
	//"fmt"
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
}

type layer struct {
	activ    Activation
	input    blas.Matrix // Z matrix of value at each node [samples, nin+1]
	output   blas.Matrix // return value at each node [samples, nout]
	weights  blas.Matrix // W outgoing weight matrix  [nout, nin+1]
	gradient blas.Matrix // gradient of weight matrix [nout, nin+1]
	deriv    blas.Matrix // Fp matrix of derivative of activation fn [samples, nin]
	delta    blas.Matrix // D matrix of errors at each node [samples, nout]
}

// AddLayer method adds a new input or hidden layer to the network.
func (n *Network) AddLayer(nin, nout int, a Activation) {
	batch := n.BatchSize
	l := &layer{
		activ:    a,
		input:    blas.New(batch, nin+1),
		output:   blas.New(batch, nout),
		weights:  blas.New(nout, nin+1),
		gradient: blas.New(nout, nin+1),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nin)
	}
	if n.Layers > 0 {
		l.delta = blas.New(batch, nin)
	}
	n.Add(l)
}

func (l *layer) Weights() blas.Matrix { return l.weights }

func (l *layer) Gradient() blas.Matrix { return l.gradient }

func (l *layer) Cost(t blas.Matrix) float64 { panic("no cost for input or hidden layer!") }

func (l *layer) FeedForward(in blas.Matrix) blas.Matrix {
	l.activ.Func.Apply(in, l.input)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(in, l.deriv)
	}
	nin := in.Cols()
	l.input.Reshape(in.Rows(), nin+1, false)
	l.input.Col(nin, nin+1).Load(blas.ColMajor, 1)
	l.output.Mul(l.input, l.weights, false, true, false)
	return l.output
}

func (l *layer) BackProp(err blas.Matrix, eta float64) blas.Matrix {
	// update gradient
	l.gradient.Mul(err, l.input, true, false, false)  // [samples, nout].T x [samples, nin+1]
	l.gradient.Scale(-eta / float64(l.output.Rows())) // [nout, nin+1]
	// propagate error backward
	if l.delta != nil {
		c := l.weights.Cols()
		l.delta.Mul(l.weights.Col(0, c-1), err, true, true, true) // [nout, nin].T x [samples, nout].T
		l.delta.MulElem(l.delta, l.deriv)                         // [samples, nin]
	}
	return l.delta
}

type outLayer struct {
	activ  Activation
	values blas.Matrix // Z matrix of values at each node [samples, nodes]
	delta  blas.Matrix // D matrix of errors at each node [nodes, samples]
	deriv  blas.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	temp   blas.Matrix
	nin    int
}

func newOutLayer(batch, nodes int, a Activation) *outLayer {
	l := &outLayer{
		activ:  a,
		values: blas.New(batch, nodes),
		delta:  blas.New(batch, nodes),
		temp:   blas.New(batch, nodes),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nodes)
	}
	return l
}

func (l *outLayer) Weights() blas.Matrix { panic("no weights for output layer!") }

func (l *outLayer) Gradient() blas.Matrix { panic("no gradient for output layer!") }

func (l *outLayer) FeedForward(in blas.Matrix) blas.Matrix {
	l.activ.Func.Apply(in, l.values)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(in, l.deriv)
	}
	return l.values
}

func (l *outLayer) BackProp(target blas.Matrix, eta float64) blas.Matrix {
	l.delta.Add(l.values, target, -1)
	if l.activ.Deriv != nil {
		l.delta.MulElem(l.delta, l.deriv)
	}
	return l.delta
}

// QuadraticOutput method appends a quadratic cost output layer to the network.
func (n *Network) QuadraticOutput(nodes int, a Activation) {
	n.Add(&quadraticOutput{newOutLayer(n.BatchSize, nodes, a)})
}

type quadraticOutput struct{ *outLayer }

var quadCost = blas.Binary64(func(x, y float64) float64 {
	return (x - y) * (x - y)
})

func (l *quadraticOutput) Cost(target blas.Matrix) float64 {
	quadCost.Apply(l.values, target, l.temp)
	return l.temp.Sum() / float64(target.Rows())
}

// CrossEntropyOutput method appends a cross entropy output layer with softmax activation to the network.
func (n *Network) CrossEntropyOutput(nodes int) {
	n.Add(&crossEntropy{newOutLayer(n.BatchSize, nodes, Softmax)})
}

type crossEntropy struct{ *outLayer }

var xentCost = blas.Binary64(func(out, tgt float64) float64 {
	if out < 1e-10 {
		out = 1e-10
	}
	return -tgt * math.Log(out)
})

func (l *crossEntropy) Cost(target blas.Matrix) float64 {
	xentCost.Apply(l.values, target, l.temp)
	return l.temp.Sum() / float64(target.Rows())
}
