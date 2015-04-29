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
func (n *Network) Run(input *m32.Matrix) *m32.Matrix {
	output := n.Nodes[0].FeedForward(input)
	for _, layer := range n.Nodes[1:] {
		output = layer.FeedForward(output)
	}
	return output
}

// GetError method calculates the error and classification error given a set of inputs and target outputs.
func (n *Network) GetError(input, target, targetClass *m32.Matrix) (totalError, classError float32) {
	output := n.Run(input)
	totalError = m32.SumDiff2(target, output) / float32(input.Rows*output.Cols)
	n.classes.MaxCol(output)
	classError = m32.CountDiff(n.classes, targetClass, epsilon) / float32(input.Rows)
	return
}

// Train method trains the network on the given training set and updates the stats.
// stop callback function returns true if we should terminate the run.
func (n *Network) Train(t data.Dataset, learnRate float32, s *Stats, stop func(int) bool) int {
	// reset stats
	s.Clear()
	start := time.Now()
	for {
		// update weights
		// TODO: add hidden nodes
		layer := n.Nodes[0]
		layer.BackProp(t.Train.Input, t.Train.Output, learnRate)

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
