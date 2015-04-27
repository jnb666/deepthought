package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/m32"
	"math"
)

// Stats struct has matrix with error on each set over time
type Stats struct {
	Test  StatsData
	Train StatsData
	Valid StatsData
}

// NewStats function initialises the stats matrices
func NewStats(maxEpoch int) Stats {
	return Stats{
		StatsData{m32.NewVector(0, maxEpoch), m32.NewVector(0, maxEpoch)},
		StatsData{m32.NewVector(0, maxEpoch), m32.NewVector(0, maxEpoch)},
		StatsData{m32.NewVector(0, maxEpoch), m32.NewVector(0, maxEpoch)},
	}
}

// Clear method resets the stats vectors
func (s Stats) Clear() {
	s.Test.Clear()
	s.Train.Clear()
	s.Valid.Clear()
}

// StatsData stores vectors with the errors and classification errors
type StatsData struct {
	Error      *m32.Matrix
	ClassError *m32.Matrix
}

// Clear method resets the stats vectors
func (d StatsData) Clear() {
	d.Error.Rows = 0
	d.ClassError.Rows = 0
}

// Print method formats the stats for printing
func (d StatsData) Print(epoch int) string {
	return fmt.Sprintf("%.3f %4.1f%%", d.Error.At(epoch, 0), 100*d.ClassError.At(epoch, 0))
}

// Update method calculates the error and updates the stats.
func (s *StatsData) Update(n *Network, epoch int, d data.Data) {
	totalError, classError := n.GetError(d.Input, d.Output, d.Classes)
	s.Error.Append(totalError)
	s.ClassError.Append(classError)
}

// Running mean and stddev as per http://www.johndcook.com/blog/standard_deviation/
type RunningStat struct {
	Count, Max, Last  float64
	Mean, Var, StdDev float64
	oldM, oldV        float64
}

func (s *RunningStat) Push(x float64) {
	s.Last = x
	if x > s.Max {
		s.Max = x
	}
	s.Count++
	if s.Count == 1 {
		s.oldM, s.Mean = x, x
	} else {
		s.Mean = s.oldM + (x-s.oldM)/s.Count
		s.Var = s.oldV + (x-s.oldM)*(x-s.Mean)
		s.oldM, s.oldV = s.Mean, s.Var
		if s.Count > 1 {
			s.StdDev = math.Sqrt(s.Var / (s.Count - 1))
		}
	}
}

func (s *RunningStat) String() string {
	return fmt.Sprintf("mean = %.3f  std dev = %.3f  max = %.3f", s.Mean, s.StdDev, s.Max)
}
