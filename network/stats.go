package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
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
func NewStats(nepoch, nruns int, maxError, maxClsErr float64) *Stats {
	return &Stats{
		Epoch:    1,
		Test:     newStatsData(nepoch, maxError, maxClsErr),
		Train:    newStatsData(nepoch, maxError, maxClsErr),
		Valid:    newStatsData(nepoch, maxError, maxClsErr),
		RunTime:  mplot.NewStatsVector(nruns, 0, 1),
		RegError: mplot.NewStatsVector(nruns, 0, maxError),
		ClsError: mplot.NewStatsVector(nruns, 0, maxClsErr),
	}
}

func newStatsData(nepoch int, ymax1, ymax2 float64) StatsData {
	return StatsData{
		Error:         mplot.NewVector(nepoch, 0, ymax1),
		ClassError:    mplot.NewVector(nepoch, 0, ymax2),
		AvgError:      mplot.NewVector(nepoch, 0, ymax1),
		AvgClassError: mplot.NewVector(nepoch, 0, ymax2),
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
	s.Train.update(n, s.Epoch-1, d.Train)
	s.Test.update(n, s.Epoch-1, d.Test)
	s.Valid.update(n, s.Epoch-1, d.Valid)
}

func (s StatsData) update(n *Network, ix int, d data.Data) {
	totalError, classError := n.GetError(d.Input, d.Output, d.Classes)
	s.Error.Set(ix, float64(totalError))
	s.ClassError.Set(ix, float64(classError))
	s.AvgError.Set(ix, float64(totalError))
	s.AvgClassError.Set(ix, float64(classError))
}
