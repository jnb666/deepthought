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
}

// NewVector creates a new empty vector with the given capacity and y range
func NewVector(capacity int) *Vector {
	return &Vector{
		data:    make([]float64, capacity),
		samples: make([]int, capacity),
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
func (v *Vector) Set(i int, val float32) {
	if v.samples[i] == 0 {
		v.data[i] = float64(val)
	} else {
		n := float64(v.samples[i])
		v.data[i] = (float64(val) + n*v.data[i]) / (n + 1)
	}
	v.samples[i]++
	if i+1 > v.size {
		v.size = i + 1
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

func (v *Vector) String() string {
	return fmt.Sprintf("cap=%d %v", len(v.data), v.data[:v.size])
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
