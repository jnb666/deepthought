package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
)

const (
	ymin = 0.0
	ymax = 0.1
)

// Stats struct has matrix with error on each set over time
type Stats struct {
	Epoch    int
	Test     StatsData
	Train    StatsData
	Valid    StatsData
	RunTime  mplot.StatsVector
	RegError mplot.StatsVector
	ClsError mplot.StatsVector
}

// StatsData stores vectors with the errors and classification errors
type StatsData struct {
	Error         *mplot.Vector
	ClassError    *mplot.Vector
	AvgError      *mplot.Vector
	AvgClassError *mplot.Vector
}

// NewStats function initialises the stats matrices
func NewStats(nepoch, nruns int) *Stats {
	return &Stats{
		Epoch:    1,
		Test:     newStatsData(nepoch),
		Train:    newStatsData(nepoch),
		Valid:    newStatsData(nepoch),
		RunTime:  mplot.NewStatsVector(nruns, ymin, ymax),
		RegError: mplot.NewStatsVector(nruns, ymin, ymax),
		ClsError: mplot.NewStatsVector(nruns, ymin, ymax),
	}
}

func newStatsData(nepoch int) StatsData {
	return StatsData{
		Error:         mplot.NewVector(nepoch, ymin, ymax),
		ClassError:    mplot.NewVector(nepoch, ymin, ymax),
		AvgError:      mplot.NewVector(nepoch, ymin, ymax),
		AvgClassError: mplot.NewVector(nepoch, ymin, ymax),
	}
}

// Clear method resets the stats vectors
func (s *Stats) Clear() {
	s.Epoch = 1
	s.Test.clear()
	s.Train.clear()
	s.Valid.clear()
}

func (d StatsData) clear() {
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
func (s *Stats) Update(n *Network, d data.Dataset) {
	if d.Train != nil {
		s.Train.update(n, s.Epoch-1, d.Train)
	}
	if d.Test != nil {
		s.Test.update(n, s.Epoch-1, d.Test)
	}
	if d.Valid != nil {
		s.Valid.update(n, s.Epoch-1, d.Valid)
	}
}

func (s StatsData) update(n *Network, ix int, d *data.Data) {
	totalError, classError := n.GetError(d.Input, d.Output, d.Classes)
	s.Error.Set(ix, float64(totalError))
	s.ClassError.Set(ix, float64(classError))
	s.AvgError.Set(ix, float64(totalError))
	s.AvgClassError.Set(ix, float64(classError))
}
