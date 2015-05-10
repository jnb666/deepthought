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
	output   blas.Matrix // return value at each node [samples, nout]
	weights  blas.Matrix // W outgoing weight matrix  [nout, nin+1]
	gradient blas.Matrix // gradient of weight matrix [nout, nin+1]
	deriv    blas.Matrix // Fp matrix of derivative of activation fn [samples, nin]
	delta    blas.Matrix // D matrix of errors at each node [samples, nout]
	deltaT   blas.Matrix // Dt transposed error [nout, samples]
}

// AddLayer method adds a new input or hidden layer to the network.
func (n *Network) AddLayer(nin, nout int, a Activation) {
	batch := n.BatchSize
	l := &layer{
		nlayer:   n.Layers,
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
		l.deltaT = blas.New(nin, batch)
	}
	n.Add(l)
}

func (l *layer) Release() {
	l.input.Release()
	l.output.Release()
	l.weights.Release()
	l.gradient.Release()
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
	l.output.Mul(l.input, l.weights, false, true)
	if Debug {
		fmt.Printf("Layer %d feedforward: output\n%s\n", l.nlayer, l.output)
	}
	return l.output
}

func (l *layer) BackProp(err blas.Matrix, eta float64) blas.Matrix {
	// update gradient
	l.gradient.Mul(err, l.input, true, false)         // [samples, nout].T x [samples, nin+1]
	l.gradient.Scale(-eta / float64(l.output.Rows())) // [nout, nin+1]
	if Debug {
		fmt.Printf("Layer %d backprop: gradient\n%s\n", l.nlayer, l.gradient)
	}
	// propagate error backward
	if l.delta != nil {
		c := l.weights.Cols()
		l.deltaT.Mul(l.weights.Col(0, c-1), err, true, true) // [nout, nin].T x [samples, nout].T
		l.delta.MulElem(l.deltaT, l.deriv, true, false)      // [samples, nin]
		if Debug {
			fmt.Printf("Layer %d backprop: delta\n%s\n", l.nlayer, l.delta)
		}
	}
	return l.delta
}

type outLayer struct {
	activ  Activation
	values blas.Matrix // Z matrix of values at each node [samples, nodes]
	delta  blas.Matrix // D matrix of errors at each node [nodes, samples]
	deriv  blas.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	temp   blas.Matrix
	cost   blas.BinaryFunction
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
		l.delta.MulElem(l.delta, l.deriv, false, false)
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
