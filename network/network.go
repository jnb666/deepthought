package network

import (
	"fmt"
	"github.com/jnb666/deepthought/blas"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
	"math"
	"math/rand"
	"strings"
	"time"
)

const (
	epsilon = 1e-4
)

var (
	Debug = false
)

// Standard activation functions
var (
	Linear  = Activation{linear{}, nil}
	Sigmoid Activation
	Tanh    Activation
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
			Func:  blas.NewUnaryCL("1.f / (1.f + exp(-x))"),
			Deriv: blas.NewUnaryCL("x * (1.f - x)"),
		}
		Tanh = Activation{
			Func:  blas.NewUnaryCL("tanh(x)"),
			Deriv: blas.NewUnaryCL("1.f - x*x"),
		}
		Softmax = Activation{
			Func: softmax{fn: blas.NewUnaryCL("exp(x)")},
		}
	} else {
		Sigmoid = Activation{
			Func:  blas.Unary64(func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }),
			Deriv: blas.Unary64(func(x float64) float64 { return x * (1 - x) }),
		}
		Tanh = Activation{
			Func:  blas.Unary64(math.Tanh),
			Deriv: blas.Unary64(func(x float64) float64 { return 1 - x*x }),
		}
		Softmax = Activation{
			Func: softmax{fn: blas.Unary64(math.Exp)},
		}
	}
}

type linear struct{}

func (linear) Apply(x, y blas.Matrix) blas.Matrix { return y.Copy(x) }

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
	testBatches  int
}

// NewNetwork function initialises a new network, samples is the maximum number of samples, i.e. minibatch size.
func NewNetwork(samples int) *Network {
	return &Network{
		BatchSize:   samples,
		classes:     blas.New(samples, 1),
		testBatches: 1,
	}
}

// Add method appends a new layer to the network
func (n *Network) Add(l Layer) {
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
	}
}

// FeedForward method calculates output from the network given input
func (n *Network) FeedForward(m blas.Matrix) blas.Matrix {
	for _, layer := range n.Nodes {
		m = layer.FeedForward(m)
	}
	return m
}

// Set number of batches to use when evaluating the error
func (n *Network) TestBatches(num int) {
	n.testBatches = num
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(d *data.Data) (totalErr, classErr float64) {
	costFn := n.Nodes[n.Layers-1].Cost
	outputs := float64(d.Output[0].Cols())
	batchSize := float64(d.Output[0].Rows())
	// calc average over a random sample of test batches
	totalError := new(mplot.RunningStat)
	classError := new(mplot.RunningStat)
	for i, batch := range rand.Perm(n.testBatches) {
		if n.Verbose && i > 0 && i%10 == 0 {
			fmt.Printf("\rtest batch: %d/%d        ", i+1, n.testBatches)
		}
		// get cost
		output := n.FeedForward(d.Input[batch])
		totalError.Push(costFn(d.Output[batch]) / outputs)
		// get classification error
		n.out2class.Apply(output, n.classes)
		n.classes.Cmp(n.classes, d.Classes[batch], epsilon)
		classError.Push(n.classes.Sum() / batchSize)
	}
	if n.testBatches > 1 {
		fmt.Print("\r")
	}
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
			cost1 := output.Cost(target) * n.checkScale
			weight[ix] = val + epsilon
			w.Load(blas.RowMajor, weight...)
			n.FeedForward(input)
			cost2 := output.Cost(target) * n.checkScale
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

// Train step method performs one training step
func (n *Network) TrainStep(in, out blas.Matrix, learnRate float64, s *Stats, batch int) {
	n.FeedForward(in)
	delta := out
	for i := n.Layers - 1; i >= 0; i-- {
		delta = n.Nodes[i].BackProp(delta, learnRate)
	}
	// check gradients?
	if batch == 0 && n.checkEvery > 0 && s.Epoch%n.checkEvery == 0 {
		n.doCheck(in, out)
	}
	for _, layer := range n.Nodes[:n.Layers-1] {
		layer.Weights().Add(layer.Weights(), layer.Gradient(), 1)
	}
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(d *data.Dataset, learnRate float64, s *Stats, stop func(*Stats) bool) int {
	n.out2class = d.OutputToClass
	s.StartRun()
	for {
		s.StartEpoch = time.Now()
		// train over each batch of data - present batches in random order
		nbatch := len(d.Train.Input)
		if nbatch > 1 {
			for i, batch := range rand.Perm(nbatch) {
				if n.Verbose && i > 0 && i%5 == 0 {
					fmt.Printf("\rtrain batch: %d/%d        ", i+1, nbatch)
				}
				n.TrainStep(d.Train.Input[batch], d.Train.Output[batch], learnRate, s, batch)
			}
		} else {
			n.TrainStep(d.Train.Input[0], d.Train.Output[0], learnRate, s, 0)
		}
		// update stats and check stopping condition
		s.Update(n, d)
		if stop(s) {
			break
		}
		s.Epoch++

	}
	s.EndRun()
	return s.Epoch
}

// SeedRandom function sets the random seed, or seeds using time if input is zero
func SeedRandom(seed int64) {
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	fmt.Println("random seed =", seed)
	rand.Seed(seed)
}
