package network

import (
	"fmt"
	"github.com/jnb666/deepthought/data"
	"github.com/jnb666/deepthought/mplot"
	"time"
)

const nbins = 100

// Stats struct has matrix with error on each set over time
type Stats struct {
	Epoch      int
	StartTime  time.Time
	StartEpoch time.Time
	Test       StatsData
	Train      StatsData
	Valid      StatsData
	NumEpochs  mplot.StatsVector
	RunTime    mplot.StatsVector
	RegError   mplot.StatsVector
	ClsError   mplot.StatsVector
}

// StatsData stores vectors with the errors and classification errors
type StatsData struct {
	Error         *mplot.Vector
	ClassError    *mplot.Vector
	AvgError      *mplot.Vector
	AvgClassError *mplot.Vector
	ErrorHist     *mplot.Histogram
}

// NewStats function initialises the stats matrices
func NewStats(nepoch, nruns int, ymax float64) *Stats {
	return &Stats{
		Epoch:     1,
		Test:      newStatsData(nepoch, ymax),
		Train:     newStatsData(nepoch, ymax),
		Valid:     newStatsData(nepoch, ymax),
		NumEpochs: mplot.NewStatsVector(nruns),
		RunTime:   mplot.NewStatsVector(nruns),
		RegError:  mplot.NewStatsVector(nruns),
		ClsError:  mplot.NewStatsVector(nruns),
	}
}

func newStatsData(nepoch int, ymax float64) StatsData {
	return StatsData{
		Error:         mplot.NewVector(nepoch),
		ClassError:    mplot.NewVector(nepoch),
		AvgError:      mplot.NewVector(nepoch),
		AvgClassError: mplot.NewVector(nepoch),
		ErrorHist:     mplot.NewHistogram(nbins, 0, ymax),
	}
}

// StartRun method resets the stats vectors for this run and starts the timer.
func (s *Stats) StartRun() {
	s.Epoch = 1
	s.Test.clear()
	s.Train.clear()
	s.Valid.clear()
	s.StartTime = time.Now()
}

// EndRun method updates per run statistics.
func (s *Stats) EndRun() int {
	s.NumEpochs.Push(float64(s.Epoch + 1))
	s.RunTime.Push(time.Since(s.StartTime).Seconds())
	test := s.Test
	if test.Error.Len() == 0 {
		test = s.Train
	}
	s.RegError.Push(test.Error.Last())
	s.ClsError.Push(test.ClassError.Last())
	return s.Epoch
}

func (d StatsData) clear() {
	d.Error.Clear()
	d.ClassError.Clear()
}

// String method prints the stats for logging.
func (s *Stats) String() string {
	str := fmt.Sprintf("%4d:", s.Epoch)
	if s.Train.Error.Len() > 0 {
		str += fmt.Sprint("   train ", s.Train)
	}
	if s.Valid.Error.Len() > 0 {
		str += fmt.Sprint("   valid ", s.Valid)
	}
	if s.Test.Error.Len() > 0 {
		str += fmt.Sprint("   test ", s.Test)
	}
	str += fmt.Sprintf("   time %.3gs", time.Since(s.StartEpoch).Seconds())
	return str
}

func (d StatsData) String() string {
	return fmt.Sprintf("%.5f %4.1f%%", d.Error.Last(), 100*d.ClassError.Last())
}

// Update method calculates the error and updates the stats.
func (s *Stats) Update(n *Network, d *data.Dataset) {
	samples := s.Valid.update(n, s.Epoch-1, d.Valid, 0)
	samples = s.Test.update(n, s.Epoch-1, d.Test, samples)
	s.Train.update(n, s.Epoch-1, d.Train, samples)
}

func (s StatsData) update(n *Network, ix int, d *data.Data, samples int) int {
	if d == nil {
		return samples
	}
	if samples == 0 || samples > d.NumSamples {
		samples = d.NumSamples
	}
	totalError, classError := n.GetError(samples, d, s.ErrorHist)
	s.Error.Set(ix, totalError)
	s.ClassError.Set(ix, classError)
	s.AvgError.Set(ix, totalError)
	s.AvgClassError.Set(ix, classError)
	return samples
}

// ErrorPlots method returns line plots for each error curve
func (s *Stats) ErrorPlots(d *data.Dataset) (p1, p2 []*mplot.Line) {
	if d.Train != nil {
		p1 = append(p1, mplot.NewLine(s.Train.Error, "training"))
		p2 = append(p2, mplot.NewLine(s.Train.ClassError, "training"))
	}
	if d.Valid != nil {
		p1 = append(p1, mplot.NewLine(s.Valid.Error, "validation"))
		p2 = append(p2, mplot.NewLine(s.Valid.ClassError, "validation"))
	}
	if d.Test != nil {
		p1 = append(p1, mplot.NewLine(s.Test.Error, "test set"))
		p2 = append(p2, mplot.NewLine(s.Test.ClassError, "test set"))
	}
	return
}

// Release resources
func (s *Stats) Release() {
	s.Train.ErrorHist.Release()
	s.Test.ErrorHist.Release()
	s.Valid.ErrorHist.Release()
}
