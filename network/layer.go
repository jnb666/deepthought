package network

import (
	"github.com/jnb666/deepthought/blas"
	"math"
)

// Layer interface type represents one layer in the network.
type Layer interface {
	Dims() []int
	Values() blas.Matrix
	FeedForward(in blas.Matrix) blas.Matrix
	BackProp(err blas.Matrix, momentum float32) blas.Matrix
	Weights() blas.Matrix
	Gradient() blas.Matrix
	Cost(t blas.Matrix) blas.Matrix
	Release()
}

type layer struct {
	nlayer    int
	dims      []int
	activ     Activation
	input     blas.Matrix // Z matrix of value at each node [samples, nin+1]
	output    blas.Matrix // return value at each node [samples, nout]
	weights   blas.Matrix // W outgoing weight matrix  [nout, nin+1]
	gradient  blas.Matrix // G gradient of weight matrix [nout, nin+1]
	gradient2 blas.Matrix // G' gradient of weight matrix [nout, nin+1]
	deriv     blas.Matrix // Fp matrix of derivative of activation fn [samples, nin]
	delta     blas.Matrix // D matrix of errors at each node [samples, nin]
}

// AddLayer method adds a new input or hidden layer to the network.
func (n *Network) AddLayer(dims []int, nout int, a Activation) {
	batch := n.BatchSize
	nin := 1
	for _, n := range dims {
		nin *= n
	}
	l := &layer{
		nlayer:    n.Layers,
		dims:      dims,
		activ:     a,
		input:     blas.New(batch, nin+1),
		output:    blas.New(batch, nout),
		weights:   blas.New(nout, nin+1),
		gradient:  blas.New(nout, nin+1),
		gradient2: blas.New(nout, nin+1),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nin)
	}
	if n.Layers > 0 {
		l.delta = blas.New(nin, batch)
	}
	n.add(l)
}

func (l *layer) Dims() []int {
	return l.dims
}

func (l *layer) Values() blas.Matrix {
	return l.input
}

func (l *layer) Release() {
	l.input.Release()
	l.output.Release()
	l.weights.Release()
	l.gradient.Release()
	if l.deriv != nil {
		l.deriv.Release()
	}
	if l.delta != nil {
		l.delta.Release()
	}
}

func (l *layer) Weights() blas.Matrix { return l.weights }

func (l *layer) Gradient() blas.Matrix { return l.gradient }

func (l *layer) Cost(t blas.Matrix) blas.Matrix { panic("no cost for input or hidden layer!") }

func (l *layer) FeedForward(in blas.Matrix) blas.Matrix {
	l.activ.Func.Apply(in, l.input)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(in, l.deriv)
	}
	nin := in.Cols()
	l.input.Reshape(in.Rows(), nin+1, false)
	l.input.Col(nin, nin+1).Set(1)
	l.output.Mul(l.input, l.weights, false, true, false)
	return l.output
}

func (l *layer) BackProp(err blas.Matrix, momentum float32) blas.Matrix {
	// calculate the gradient
	if momentum == 0 {
		l.gradient.Mul(err, l.input, true, false, false)
	} else {
		l.gradient2.Mul(err, l.input, true, false, false)
		l.gradient.Add(l.gradient2, l.gradient, momentum)
	}
	// propagate error backward
	if l.delta != nil {
		c := l.weights.Cols()
		l.delta.Mul(l.weights.Col(0, c-1), err, true, true, true)
		l.delta.MulElem(l.delta, l.deriv)
	}
	return l.delta
}

type outLayer struct {
	activ  Activation
	dims   []int
	values blas.Matrix // Z matrix of values at each node [samples, nodes]
	delta  blas.Matrix // D matrix of errors at each node [samples, nodes]
	deriv  blas.Matrix // Fp matrix of derivative of activation fn [nin, samples]
	costs  blas.Matrix // cost for each sample in data set [samples, 1]
	temp   blas.Matrix
	cost   blas.BinaryFunction
}

func newOutLayer(batch, nodes int, a Activation) *outLayer {
	l := &outLayer{
		activ:  a,
		dims:   []int{nodes},
		values: blas.New(batch, nodes),
		delta:  blas.New(batch, nodes),
		costs:  blas.New(batch, 1),
		temp:   blas.New(batch, nodes),
	}
	if a.Deriv != nil {
		l.deriv = blas.New(batch, nodes)
	}
	return l
}

func (l *outLayer) Dims() []int {
	return l.dims
}

func (l *outLayer) Values() blas.Matrix {
	return l.values
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
	l.activ.Func.Apply(in, l.values)
	if l.activ.Deriv != nil {
		l.activ.Deriv.Apply(in, l.deriv)
	}
	return l.values
}

func (l *outLayer) BackProp(target blas.Matrix, momentum float32) blas.Matrix {
	l.delta.Add(l.values, target, -1)
	if l.activ.Deriv != nil {
		l.delta.MulElem(l.delta, l.deriv)
	}
	return l.delta
}

func (l *outLayer) Cost(target blas.Matrix) blas.Matrix {
	l.cost.Apply(l.values, target, l.temp)
	return l.costs.SumRows(l.temp)
}

// AddQuadraticOutput method appends a quadratic cost output layer to the network.
func (n *Network) AddQuadraticOutput(nodes int, a Activation) {
	layer := newOutLayer(n.BatchSize, nodes, a)
	if blas.Implementation() == blas.OpenCL32 {
		layer.cost = blas.NewBinaryCL("float z = (x-y)*(x-y);")
	} else {
		layer.cost = blas.Binary32(func(out, tgt float32) float32 { return (out - tgt) * (out - tgt) })
	}
	n.add(layer)
}

// AddCrossEntropyOutput method appends a cross entropy output layer with softmax activation to the network.
func (n *Network) AddCrossEntropyOutput(nodes int) {
	layer := newOutLayer(n.BatchSize, nodes, Softmax)
	if blas.Implementation() == blas.OpenCL32 {
		layer.cost = blas.NewBinaryCL("float z = -y * log(max(x, 1e-10f));")
	} else {
		layer.cost = blas.Binary32(func(out, tgt float32) float32 {
			if out < 1e-10 {
				out = 1e-10
			}
			return -tgt * float32(math.Log(float64(out)))
		})
	}
	n.add(layer)
}
