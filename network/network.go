package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
	"math/rand"
	"strings"
	"time"
)

// Neural network type is an array of layers.
type Network struct {
	Nodes     []Layer
	BatchSize int
	classes   *m32.Matrix
	out2class func(a, b *m32.Matrix)
}

// NewNetwork function initialises a new network
func NewNetwork(batchSize int) *Network {
	return &Network{
		Nodes:     []Layer{},
		BatchSize: batchSize,
		classes:   m32.New(batchSize, 1),
	}
}

// Add method appends a new layer to the network
func (n *Network) Add(l Layer) {
	n.Nodes = append(n.Nodes, l)
}

// String method returns a printable representation of the network.
func (n *Network) String() string {
	str := make([]string, len(n.Nodes))
	for i, layer := range n.Nodes {
		str[i] = layer.String()
	}
	return strings.Join(str, "\n")
}

// SetRandomWeights method initalises the weights to random values from -max to +max.
func (n *Network) SetRandomWeights(max float32) {
	for _, l := range n.Nodes {
		l.Weights().Random(-max, max)
	}
}

// Run method calculates output from the network given input
func (n *Network) FeedForward(input *m32.Matrix) *m32.Matrix {
	output := n.Nodes[0].FeedForward(input)
	for _, layer := range n.Nodes[1:] {
		output = layer.FeedForward(output)
	}
	return output
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(input, target, targetClass *m32.Matrix) (totalError, classError float32) {
	output := n.FeedForward(input)
	totalError = m32.SumDiff2(target, output) / float32(input.Rows*output.Cols)
	n.out2class(output, n.classes)
	classError = m32.CountDiff(n.classes, targetClass, epsilon) / float32(input.Rows)
	return
}

// Errors method returns the error matrix for the ith layer, or nil of layer does not exist
func (n *Network) Errors(i int) *m32.Matrix {
	if i >= 0 && i < len(n.Nodes) {
		return n.Nodes[i].Errors()
	}
	return nil
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(t data.Dataset, learnRate float32, s *Stats, stop func(int) bool) int {
	n.out2class = t.OutputToClass
	// reset stats
	s.Clear()
	start := time.Now()
	last := len(n.Nodes) - 1
	for {
		// forward propagate input
		output := n.FeedForward(t.Train.Input)

		// errors at output
		n.Errors(last).Add(-1, output, t.Train.Output)

		// back propagate errors
		for i := last; i >= 0; i-- {
			n.Nodes[i].BackProp(learnRate, n.Errors(i-1))
		}

		// update stats
		s.Update(n, t)

		// next epoch
		if stop(s.Epoch) {
			break
		}
		s.Epoch++
	}
	// per run stats
	s.RunTime.Push(time.Since(start).Seconds())
	s.RegError.Push(s.Test.Error.Last())
	s.ClsError.Push(s.Test.ClassError.Last())
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
