// Package network implements feedforward neural nets with training using stochastic gradient descent.
package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/vec"
	"math"
	"math/rand"
	"strings"
	"time"
)

const epsilon = 1e-4

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
			Func: blas.Unary64(sigmoid),
			Deriv: blas.Unary64(func(x float64) float64 {
				y := sigmoid(x)
				return y * (1 - y)
			}),
		}
		Tanh = Activation{
			Func: blas.Unary64(math.Tanh),
			Deriv: blas.Unary64(func(x float64) float64 {
				y := math.Tanh(x)
				return 1 - y*y
			}),
		}
		Relu = Activation{
			Func: blas.Unary64(func(x float64) float64 {
				return math.Max(x, 0)
			}),
			Deriv: blas.Unary64(func(x float64) float64 {
				if x >= 0 {
					return 1
				}
				return 0
			}),
		}
		Softmax = Activation{
			Func: softmax{fn: blas.Unary64(math.Exp)},
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
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
	checkMax     float64
	checkScale   float64
	input        blas.Matrix
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
		n.output.Release()
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

// SetRandomWeights method initalises the weights to random values.
// Uses a normal distribution with mean zero and std dev 1/sqrt(num_inputs) for the weights.
// Bias weights are left at zero.
func (n *Network) SetRandomWeights() {
	for _, layer := range n.Nodes[:n.Layers-1] {
		w := layer.Weights()
		nin, nout := w.Cols()-1, w.Rows()
		data := make([]float64, (nin+1)*nout)
		for i := range data[:nin*nout] {
			data[i] = rand.NormFloat64() / math.Sqrt(float64(nin))
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

// GetError method calculates the error and classification error given a set of inputs and target outputs.
// samples parameter is the maximum number of samples to check.
func (n *Network) GetError(samples int, d *data.Data, hist *vec.Vector, hmax float64) (totalErr, classErr float64) {
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
		totalError.Push(cost.Sum() / float64(rows*cols))
		// get classification error
		n.out2class.Apply(output, n.classes)
		n.classes.Cmp(n.classes, d.Classes.Row(ix, ix+rows), epsilon)
		classError.Push(n.classes.Sum() / float64(rows))
		if n.Verbose {
			fmt.Printf("\rtest batch: %d/%d        ", ix+rows, samples)
		}
	}
	if n.Verbose {
		fmt.Print("\r")
	}
	hist.Set(0, hmax/histBins, n.errorHist.Data(blas.ColMajor))
	return totalError.Mean, classError.Mean
}

// CheckGradient method enables gradient check every nepochs runs with max error of maxError.
// if samples is > 0 then limit to this number of samples per layer.
// scale fudge factor shouldn't be needed!
func (n *Network) CheckGradient(nepochs int, maxError float64, samples int, scale float64) {
	n.checkEvery = nepochs
	n.checkMax = maxError
	n.checkSamples = samples
	n.checkScale = scale
}

func (n *Network) doCheck(input, target blas.Matrix) (ok bool) {
	output := n.Nodes[n.Layers-1]
	rows := float64(input.Rows())
	ok = true
	for nlayer, layer := range n.Nodes[:n.Layers-1] {
		w := layer.Weights()
		weight := w.Data(blas.RowMajor)
		gradData := layer.Gradient().Data(blas.RowMajor)
		nweight := len(weight)
		samples := n.checkSamples
		if samples == 0 || nweight < samples {
			samples = nweight
		}
		grads := make([]float64, samples)
		diffs := make([]float64, samples)
		projs := make([]float64, samples)
		maxDiff := 0.0
		for i, ix := range rand.Perm(nweight)[:samples] {
			fmt.Printf("\rcheck %d/%d        ", i+1, samples)
			val := weight[ix]
			weight[ix] = val - epsilon
			w.Load(blas.RowMajor, weight...)
			n.FeedForward(input)
			cost1 := output.Cost(target).Sum() * n.checkScale / rows
			weight[ix] = val + epsilon
			w.Load(blas.RowMajor, weight...)
			n.FeedForward(input)
			cost2 := output.Cost(target).Sum() * n.checkScale / rows
			grads[i] = gradData[ix]
			projs[i] = (cost1 - cost2) / (2 * epsilon)
			divisor := math.Abs(grads[i]) + math.Abs(projs[i])
			if divisor > epsilon {
				diffs[i] = math.Abs(grads[i]-projs[i]) / divisor
			}
			maxDiff = math.Max(maxDiff, diffs[i])
			weight[ix] = val
		}
		w.Load(blas.RowMajor, weight...)
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
func (n *Network) TrainStep(epoch, batch, samples int, eta, lambda, momentum float64) {
	n.FeedForward(n.input)
	// back propagate error and scale gradient
	delta := n.Nodes[n.Layers-1].BackProp(n.output, momentum)
	batchSize := float64(n.input.Rows())
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
	weightScale := 1 - eta*lambda/float64(samples)
	for _, layer := range n.Nodes[:n.Layers-1] {
		w := layer.Weights()
		if lambda != 0 {
			w.Col(0, w.Cols()-1).Scale(weightScale)
		}
		w.Add(w, layer.Gradient(), 1)
	}
}

// Train method trains the network on the given training set for one epoch.
func (n *Network) Train(s *Stats, d *data.Dataset, cfg *Config) {
	if n.input == nil {
		n.input = blas.New(n.BatchSize, d.Train.Input.Cols())
		n.output = blas.New(n.BatchSize, d.Train.Output.Cols())
	}
	s.Epoch++
	s.StartEpoch = time.Now()
	smp := Samplers[cfg.Sampler].Init(d.Train.NumSamples, n.BatchSize)
	batch := 0
	for {
		smp.Sample(d.Train.Input, n.input)
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

// SeedRandom function sets the random seed, or seeds using time if input is zero. Returns the seed which was used.
func SeedRandom(seed int64) int64 {
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("set random seed to", seed)
	rand.Seed(seed)
	return seed
}
