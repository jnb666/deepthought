package network

import (
	"fmt"
	"github.com/jnb666/deepthought/vec"
	"math"
	"time"
)

const (
	histBins = 80
	histMin  = 0.0
	histMax  = 1.0
	histAuto = 0.9
)

// Stats struct has matrix with error on each set over time
type Stats struct {
	Epoch      int
	Runs       int
	RunSuccess int
	StartEpoch time.Time
	TotalTime  time.Duration
	Test       *StatsData
	Train      *StatsData
	Valid      *StatsData
	RunTime    *vec.RunningStat
	RegError   *vec.RunningStat
	ClsError   *vec.RunningStat
}

// StatsData stores vectors with the errors and classification errors
type StatsData struct {
	Error      *vec.Vector
	ClassError *vec.Vector
	ErrorHist  *vec.Vector
	HistMax    float64
}

// NewStats function returns a new stats struct.
func NewStats() *Stats {
	return &Stats{
		Test:     newStatsData(),
		Train:    newStatsData(),
		Valid:    newStatsData(),
		RunTime:  &vec.RunningStat{},
		RegError: &vec.RunningStat{},
		ClsError: &vec.RunningStat{},
	}
}

func newStatsData() *StatsData {
	return &StatsData{
		Error:      vec.New(0),
		ClassError: vec.New(0),
		ErrorHist:  vec.New(histBins),
		HistMax:    histMax,
	}
}

// Reset method resets all the stats
func (s *Stats) Reset() {
	s.Epoch = 0
	s.TotalTime = 0
	s.Runs = 0
	s.RunSuccess = 0
	s.Test.clear(true)
	s.Train.clear(true)
	s.Valid.clear(true)
	s.RunTime.Clear()
	s.RegError.Clear()
	s.ClsError.Clear()
}

// StartRun method resets the stats vectors for this run and starts the timer.
func (s *Stats) StartRun() {
	s.Epoch = 0
	s.TotalTime = 0
	s.Test.clear(false)
	s.Train.clear(false)
	s.Valid.clear(false)
}

func (d *StatsData) clear(reset bool) {
	d.Error.Clear(reset)
	d.ClassError.Clear(reset)
	d.ErrorHist.Clear(reset)
	d.HistMax = histMax
}

// EndRun method updates per run statistics and returns the stats.
func (s *Stats) EndRun(failed bool) string {
	s.RunTime.Push(s.TotalTime.Seconds())
	test := s.Test
	if test.Error.Len() == 0 {
		test = s.Train
	}
	s.RegError.Push(test.Error.Last())
	s.ClsError.Push(test.ClassError.Last())
	var status string
	if failed {
		status = "**FAILED **"
	} else {
		status = "**SUCCESS**"
		s.RunSuccess++
	}
	s.Runs++
	status += fmt.Sprintf("  epochs=%d  run time=%.2fs  reg error=%.4f  class error=%.1f%%",
		s.Epoch, s.TotalTime.Seconds(), test.Error.Last(), 100*test.ClassError.Last())
	return status
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
	str += fmt.Sprintf("   time %dms", time.Since(s.StartEpoch).Nanoseconds()/1e6)
	return str
}

func (d *StatsData) String() string {
	return fmt.Sprintf("%.5f %4.1f%%", d.Error.Last(), 100*d.ClassError.Last())
}

// History method returns historical statistics
func (s *Stats) History() string {
	rate := 100 * float64(s.RunSuccess) / float64(s.Runs)
	return fmt.Sprintf("== success rate: %d / %d %.0f%% ==\nrun time:    %s\nreg error:   %s\nclass error: %s",
		s.RunSuccess, s.Runs, rate, s.RunTime, s.RegError, s.ClsError)
}

// Update method calculates the error and updates the stats.
func (s *Stats) Update(n *Network, d *Dataset) {
	s.TotalTime += time.Since(s.StartEpoch)
	dset := []*Data{d.Valid, d.Test, d.Train}
	stats := []*StatsData{s.Valid, s.Test, s.Train}
	samples := 0
	for i, set := range stats {
		if dset[i] != nil {
			samples = set.update(n, dset[i], samples)
		}
	}
	if histAuto >= 1 {
		return
	}
	// rescale the histograms
	var oldMax, newMax float64
	for _, set := range stats {
		oldMax = math.Max(oldMax, set.HistMax)
		if max := scaleHist(set.ErrorHist, histAuto); max > newMax {
			newMax = max
		}
	}
	niceMax := vec.Nicenum(newMax, true)
	if math.Abs(newMax-oldMax) < epsilon {
		return
	}
	//fmt.Printf("rescale hist: %.4f => %f\n", oldMax, niceMax)
	s.Train.ErrorHist.Lock()
	s.Test.ErrorHist.Lock()
	s.Valid.ErrorHist.Lock()
	for i, set := range stats {
		if dset[i] != nil {
			stats[i].HistMax = niceMax
			n.GetError(samples, dset[i], set.ErrorHist, niceMax)
		}
	}
	s.Train.ErrorHist.Unlock()
	s.Test.ErrorHist.Unlock()
	s.Valid.ErrorHist.Unlock()
}

func scaleHist(hist *vec.Vector, scale float64) float64 {
	var x, y, sum, total float64
	for i := 0; i < hist.Len(); i++ {
		_, y := hist.XY(i)
		total += y
	}
	for i := 0; i < hist.Len(); i++ {
		x, y = hist.XY(i)
		sum += y
		if sum >= total*scale {
			break
		}
	}
	return x + hist.BinWidth()
}

func (s *StatsData) update(n *Network, d *Data, samples int) int {
	if d == nil {
		return samples
	}
	if samples == 0 || samples > d.NumSamples {
		samples = d.NumSamples
	}
	s.ErrorHist.Lock()
	totalError, classError := n.GetError(samples, d, s.ErrorHist, s.HistMax)
	s.ErrorHist.Unlock()
	s.Error.Push(totalError, 0)
	s.ClassError.Push(classError, 0)
	return samples
}
