package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
)

// Stats struct has matrix with error on each set over time
type Stats struct {
	Epoch int
	Test  StatsData
	Train StatsData
	Valid StatsData
}

// StatsData stores vectors with the errors and classification errors
type StatsData struct {
	Error      *mplot.Vector
	ClassError *mplot.Vector
}

// NewStats function initialises the stats matrices
func NewStats(maxEpoch int) *Stats {
	return &Stats{
		Epoch: 1,
		Test:  StatsData{mplot.NewVector(maxEpoch), mplot.NewVector(maxEpoch)},
		Train: StatsData{mplot.NewVector(maxEpoch), mplot.NewVector(maxEpoch)},
		Valid: StatsData{mplot.NewVector(maxEpoch), mplot.NewVector(maxEpoch)},
	}
}

// Clear method resets the stats vectors
func (s *Stats) Clear() {
	s.Epoch = 1
	s.Test.Clear()
	s.Train.Clear()
	s.Valid.Clear()
}

func (d StatsData) Clear() {
	d.Error.Clear()
	d.ClassError.Clear()
}

// String method prints the stats for logging.
func (s *Stats) String() string {
	return fmt.Sprintf("%3d:  train %s   valid %s   test %s",
		s.Epoch, s.Train, s.Valid, s.Test)
}

func (d StatsData) String() string {
	return fmt.Sprintf("%.3f %4.1f%%", d.Error.Last(), 100*d.ClassError.Last())
}

// Update method calculates the error and updates the stats.
func (s StatsData) Update(n *Network, d data.Data) {
	totalError, classError := n.GetError(d.Input, d.Output, d.Classes)
	s.Error.Push(totalError)
	s.ClassError.Push(classError)
}
