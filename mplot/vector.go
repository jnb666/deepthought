package mplot

import (
	"fmt"
	"math"
)

// Vector holds a slice of float64 numbers. It implements the plotter.XYer interface
type Vector struct {
	data    []float64
	samples []int
	size    int
	min     float64
	max     float64
}

// NewVector creates a new empty vector with the given capacity and range
func NewVector(capacity int, min, max float64) *Vector {
	return &Vector{
		data:    make([]float64, capacity),
		samples: make([]int, capacity),
		min:     min,
		max:     max,
	}
}

// Clear method empies the elements fom the vector after each run
func (v *Vector) Clear() {
	v.size = 0
	for i := range v.samples {
		v.samples[i] = 0
	}
}

// Set method updates the running mean for the ith value
func (v *Vector) Set(i int, val float64) {
	if v.samples[i] == 0 {
		v.data[i] = val
	} else {
		n := float64(v.samples[i])
		v.data[i] = (val + n*v.data[i]) / (n + 1)
	}
	v.samples[i]++
	if i+1 > v.size {
		v.size = i + 1
	}
	if v.data[i] < v.min {
		v.min = v.data[i]
	}
	if v.data[i] > v.max {
		v.max = v.data[i]
	}
}

// Last method returns the last entry from the vector
func (v *Vector) Last() float64 {
	return v.data[v.size-1]
}

func (v *Vector) Len() int {
	return v.size
}

func (v *Vector) Cap() int {
	return len(v.data)
}

func (v *Vector) XY(i int) (x, y float64) {
	return float64(i), v.data[i]
}

// DataRange implements the plot.DataRanger interface.
func (v *Vector) DataRange() (xmin, xmax, ymin, ymax float64) {
	return 0, float64(v.Cap() - 1), v.min, v.max
}

func (v *Vector) String() string {
	return fmt.Sprintf("cap=%d %v", len(v.data), v.data[:v.size])
}

// Running mean and stddev as per http://www.johndcook.com/blog/standard_deviation/
type RunningStat struct {
	Count, Max        float64
	Mean, Var, StdDev float64
	oldM, oldV        float64
}

func (s *RunningStat) Push(x float64) {
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

// StatsVector type combines a vector with associated running stats
type StatsVector struct {
	*Vector
	*RunningStat
}

// NewStatsVector creates a new empty vector with the given capacity
func NewStatsVector(capacity int, min, max float64) StatsVector {
	return StatsVector{NewVector(capacity, min, max), &RunningStat{}}
}

// Push method adds a new element and updates the stats
func (s StatsVector) Push(val float64) {
	s.Set(s.Len(), val)
	s.RunningStat.Push(val)
}

func (s StatsVector) String() string {
	return s.RunningStat.String()
}
