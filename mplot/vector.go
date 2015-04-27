package mplot

import (
	"fmt"
	"math"
)

// Vector holds a slice of float64 numbers. It implements the plotter.XYer interface
type Vector struct {
	data []float64
}

// NewVector creates a new empty vector with the given capacity and y range
func NewVector(capacity int) *Vector {
	return &Vector{data: make([]float64, 0, capacity)}
}

func (v *Vector) Clear() {
	v.data = v.data[:0]
}

func (v *Vector) Push(val float32) {
	v.data = append(v.data, float64(val))
}

func (v *Vector) Last() float64 {
	return v.data[len(v.data)-1]
}

func (v *Vector) Len() int {
	return len(v.data)
}

func (v *Vector) Cap() int {
	return cap(v.data)
}

func (v *Vector) XY(i int) (x, y float64) {
	return float64(i), v.data[i]
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
