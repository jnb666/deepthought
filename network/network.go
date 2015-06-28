// Package network implements feedforward neural nets with training using stochastic gradient descent.
package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/vec"
	"math"
	"math/rand"
	"strings"
	"time"
)

const epsilon = 0.01

// Standard activation functions
var (
	Linear  = Activation{linear{}, nil}
	Sigmoid Activation
	Tanh    Activation
	Relu    Activation
	Softmax Activation
)

// Activation type represents the activation function and derivative
type Activation struct {
	Func  blas.UnaryFunction
	Deriv blas.UnaryFunction
}

// Init function initialises the package and set the matrix implementation.
func Init(imp blas.Impl) {
	blas.Init(imp)
	if imp == blas.OpenCL32 {
		Sigmoid = Activation{
			Func:  blas.NewUnaryCL("float y = 1.f/(1.f+exp(-x));"),
			Deriv: blas.NewUnaryCL("float s = 1.f/(1.f+exp(-x)); float y = s*(1.f-s);"),
		}
		Tanh = Activation{
			Func:  blas.NewUnaryCL("float y = tanh(x);"),
			Deriv: blas.NewUnaryCL("float s = tanh(x); float y = 1.f-s*s;"),
		}
		Relu = Activation{
			Func:  blas.NewUnaryCL("float y = max(x, 0.f);"),
			Deriv: blas.NewUnaryCL("float y = x >= 0.f ? 1.f : 0.f;"),
		}
		Softmax = Activation{
			Func: softmax{fn: blas.NewUnaryCL("float y = exp(x);")},
		}
	} else {
		Sigmoid = Activation{
			Func: blas.Unary32(sigmoid),
			Deriv: blas.Unary32(func(x float32) float32 {
				y := sigmoid(x)
				return y * (1 - y)
			}),
		}
		Tanh = Activation{
			Func: blas.Unary32(tanh),
			Deriv: blas.Unary32(func(x float32) float32 {
				y := tanh(x)
				return 1 - y*y
			}),
		}
		Relu = Activation{
			Func: blas.Unary32(func(x float32) float32 {
				if x >= 0 {
					return x
				}
				return 0
			}),
			Deriv: blas.Unary32(func(x float32) float32 {
				if x >= 0 {
					return 1
				}
				return 0
			}),
		}
		Softmax = Activation{
			Func: softmax{fn: blas.Unary32(func(x float32) float32 {
				return float32(math.Exp(float64(x)))
			})},
		}
	}
}

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

type linear struct{}

func (linear) Apply(x, y blas.Matrix) blas.Matrix { return y.Copy(x, nil) }

type softmax struct{ fn blas.UnaryFunction }

func (s softmax) Apply(x, y blas.Matrix) blas.Matrix {
	s.fn.Apply(x, y)
	return y.Norm(y)
}

// Neural network type is an array of layers.
type Network struct {
	Nodes        []Layer
	Layers       int
	BatchSize    int
	Verbose      bool
	classes      blas.Matrix
	out2class    blas.UnaryFunction
	checkEvery   int
	checkSamples int
	checkMax     float32
	checkScale   float32
	input        blas.Matrix
	rawInput     blas.Matrix
	output       blas.Matrix
	errorHist    blas.Matrix
}

// New function initialises a new network, samples is the maximum number of samples, i.e. minibatch size.
func New(samples int, out2class blas.UnaryFunction) *Network {
	return &Network{
		BatchSize: samples,
		classes:   blas.New(samples, 1),
		errorHist: blas.New(histBins, 1),
		out2class: out2class,
	}
}

func (n *Network) add(l Layer) {
	n.Nodes = append(n.Nodes, l)
	n.Layers++
}

// Release method frees up any resources used by the network.
func (n *Network) Release() {
	fmt.Println("release resources")
	for _, layer := range n.Nodes {
		layer.Release()
	}
	n.classes.Release()
	if n.input != nil {
		n.input.Release()
	}
	if n.output != nil {
		n.output.Release()
	}
	if n.rawInput != nil {
		n.rawInput.Release()
	}
}

// String method returns a printable representation of the network.
func (n *Network) String() string {
	str := make([]string, len(n.Nodes))
	for i, layer := range n.Nodes[:n.Layers-1] {
		str[i] = fmt.Sprintf("== Layer %d ==\n%s", i, layer.Weights())
	}
	return strings.Join(str, "\n")
}

// SetRandomWeights method initalises the weights to random values and sets the gradients to zero.
// Uses a normal distribution with mean zero and std dev 1/sqrt(num_inputs) for the weights.
// Bias weights are left at zero.
func (n *Network) SetRandomWeights() {
	for _, layer := range n.Nodes[:n.Layers-1] {
		w := layer.Weights()
		nin, nout := w.Cols()-1, w.Rows()
		data := make([]float32, (nin+1)*nout)
		for i := range data[:nin*nout] {
			data[i] = float32(rand.NormFloat64() / math.Sqrt(float64(nin)))
		}
		w.Load(blas.ColMajor, data...)
		layer.Gradient().Set(0)
	}
}

// FeedForward method calculates output from the network given input
func (n *Network) FeedForward(m blas.Matrix) blas.Matrix {
	for _, layer := range n.Nodes {
		m = layer.FeedForward(m)
	}
	return m
}

// Classify method returns a column vector with classified output
func (n *Network) Classify(output blas.Matrix) blas.Matrix {
	n.out2class.Apply(output, n.classes)
	return n.classes
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
// samples parameter is the maximum number of samples to check.
func (n *Network) GetError(samples int, d *Data, hist *vec.Vector, hmax float32) (totalErr, classErr float32) {
	totalError := new(vec.RunningStat)
	classError := new(vec.RunningStat)
	cols := d.Output.Cols()
	rows := n.BatchSize
	if rows > samples {
		rows = samples
	}
	n.errorHist.Set(0)
	for ix := 0; ix < samples; ix += rows {
		// get cost per sample
		output := n.FeedForward(d.Input.Row(ix, ix+rows))
		cost := n.Nodes[n.Layers-1].Cost(d.Output.Row(ix, ix+rows))
		n.errorHist.Histogram(cost, histBins, histMin, hmax)
		// average error over dataset
		totalError.Push(cost.Sum() / float32(rows*cols))
		// get classification error
		n.out2class.Apply(output, n.classes)
		n.classes.Cmp(n.classes, d.Classes.Row(ix, ix+rows), epsilon)
		classError.Push(n.classes.Sum() / float32(rows))
		if n.Verbose {
			fmt.Printf("\rtest batch: %d/%d        ", ix+rows, samples)
		}
	}
	if n.Verbose {
		fmt.Print("\r")
	}
	hist.Set(0, hmax/histBins, n.errorHist.Data(blas.ColMajor))
	return float32(totalError.Mean), float32(classError.Mean)
}

// CheckGradient method enables gradient check every nepochs runs with max error of maxError.
// if samples is > 0 then limit to this number of samples per layer.
// scale fudge factor shouldn't be needed!
func (n *Network) CheckGradient(nepochs int, maxError float32, samples int, scale float32) {
	n.checkEvery = nepochs
	n.checkMax = maxError
	n.checkSamples = samples
	n.checkScale = scale
}

func (n *Network) doCheck(input, target blas.Matrix) (ok bool) {
	output := n.Nodes[n.Layers-1]
	rows := float32(input.Rows())
	ok = true
	for nlayer, layer := range n.Nodes[:n.Layers-1] {
		weight := layer.Weights()
		weightData := weight.Data(blas.RowMajor)
		gradient := layer.Gradient()
		gradData := gradient.Data(blas.RowMajor)
		nweight := len(weightData)
		samples := n.checkSamples
		if samples == 0 || nweight < samples {
			samples = nweight
		}
		grads := make([]float32, samples)
		diffs := make([]float32, samples)
		projs := make([]float32, samples)
		maxDiff := float32(0)
		for i, ix := range rand.Perm(nweight)[:samples] {
			fmt.Printf("\rcheck %d/%d        ", i+1, samples)
			val := weightData[ix]
			weightData[ix] = val - epsilon
			weight.Load(blas.RowMajor, weightData...)
			n.FeedForward(input)
			cost1 := output.Cost(target).Sum() * n.checkScale / rows
			weightData[ix] = val + epsilon
			weight.Load(blas.RowMajor, weightData...)
			n.FeedForward(input)
			cost2 := output.Cost(target).Sum() * n.checkScale / rows
			grads[i] = gradData[ix]
			projs[i] = (cost1 - cost2) / (2 * epsilon)
			diffs[i] = vec.Abs(grads[i] - projs[i])
			maxDiff = vec.Max(maxDiff, diffs[i])
			weightData[ix] = val
		}
		weight.Load(blas.RowMajor, weightData...)
		gradient.Load(blas.RowMajor, gradData...)
		fmt.Printf("\rLAYER %d : max diff=%.3g epsilon=%g\n", nlayer, maxDiff, epsilon)
		if maxDiff > n.checkMax {
			grad := blas.New(1, samples).Load(blas.RowMajor, grads...)
			grad.SetFormat("%11.8f")
			proj := blas.New(1, samples).Load(blas.RowMajor, projs...)
			proj.SetFormat("%11.8f")
			diff := blas.New(1, samples).Load(blas.RowMajor, diffs...)
			diff.SetFormat("%11.8f")
			fmt.Printf("*** WARNING *** GRADIENTS LOOK WRONG!\n%s\n%s\n%s\n", grad, proj, diff)
			ok = false
			grad.Release()
			proj.Release()
			diff.Release()
		}
	}
	return
}

// Train step method performs one training step. eta is the learning rate, lambda is the weight decay.
func (n *Network) TrainStep(epoch, batch, samples int, eta, lambda, momentum float32) {
	n.FeedForward(n.input)
	// back propagate error and scale gradient
	delta := n.Nodes[n.Layers-1].BackProp(n.output, momentum)
	batchSize := float32(n.input.Rows())
	for i := n.Layers - 2; i >= 0; i-- {
		layer := n.Nodes[i]
		delta = layer.BackProp(delta, momentum)
		layer.Gradient().Scale(-eta / batchSize)
	}
	// optionally check gradients
	if batch == 0 && n.checkEvery > 0 && epoch%n.checkEvery == 0 {
		n.doCheck(n.input, n.output)
	}
	// update weights
	weightScale := 1 - eta*lambda/float32(samples)
	for _, layer := range n.Nodes[:n.Layers-1] {
		w := layer.Weights()
		if lambda != 0 {
			w.Col(0, w.Cols()-1).Scale(weightScale)
		}
		w.Add(w, layer.Gradient(), 1)
	}
}

// Train method trains the network on the given training set for one epoch.
func (n *Network) Train(s *Stats, d *Dataset, cfg *Config) {
	if n.input == nil {
		n.rawInput = blas.New(n.BatchSize, d.Train.Input.Cols())
		n.input = blas.New(n.BatchSize, d.Train.Input.Cols())
		n.output = blas.New(n.BatchSize, d.Train.Output.Cols())
	}
	s.Epoch++
	s.StartEpoch = time.Now()
	smp := NewSampler(cfg.Sampler).Init(d.Train.NumSamples, n.BatchSize)
	batch := 0
	for {
		smp.Sample(d.Train.Input, n.rawInput)
		if cfg.Distortion > 0 {
			//d.Load.Debug(batch == 0)
			d.Load.Distort(n.rawInput, n.input, -1, cfg.Distortion)
		} else {
			n.input = n.rawInput
		}
		smp.Sample(d.Train.Output, n.output)
		n.TrainStep(s.Epoch, batch, d.Train.NumSamples, cfg.LearnRate, cfg.WeightDecay, cfg.Momentum)
		batch++
		if n.Verbose {
			fmt.Printf("\rtrain batch: %d/%d        ", batch, d.Train.NumSamples/n.BatchSize)
		}
		if !smp.Next() {
			break
		}
	}
	smp.Release()
}
