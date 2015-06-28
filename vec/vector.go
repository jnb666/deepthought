// Package vector implements vector types for plotting and statistics.
package vec

import (
	"fmt"
	"math"
	"sync"
)

// Vector holds a slice of float32 values and errors. It implements the qml.XYer interface
type Vector struct {
	data        []float32
	errors      []float32
	xmin, xstep float32
	ymin, ymax  float32
	defSize     int
	sync.Mutex
}

// New function creates a new empty vector with the given capacity and range
func New(defaultSize int) *Vector {
	v := &Vector{defSize: defaultSize}
	v.Clear(true)
	return v
}

func (v *Vector) Len() int { return len(v.data) }

func (v *Vector) BinWidth() float32 { return v.xstep }

func (v *Vector) XY(i int) (x, y float32) {
	return v.xmin + v.xstep*float32(i), v.data[i]
}

func (v *Vector) XYErr(i int) (x, y, yerr float32) {
	return v.xmin + v.xstep*float32(i), v.data[i], v.errors[i]
}

func (v *Vector) DataRange() (xmin, ymin, xmax, ymax float32) {
	return v.xmin, v.ymin, v.xmin + float32(cap(v.data))*v.xstep, v.ymax
}

func (v *Vector) SetWidth(w float32) *Vector {
	v.xstep = w
	return v
}

// Clear method empies the elements fom the vector after each run
func (v *Vector) Clear(reset bool) *Vector {
	if reset {
		v.data = make([]float32, 0, v.defSize)
		v.errors = make([]float32, 0, v.defSize)
		v.ymin, v.ymax = 1e30, -1e30
		v.xstep = 1
	} else {
		v.data = v.data[:0]
	}
	return v
}

// Set method sets values from slice
func (v *Vector) Set(xmin, xstep float32, values []float32) {
	v.data = append(v.data[:0], values...)
	v.xmin = xmin
	v.xstep = xstep
	for _, val := range v.data {
		if val < v.ymin {
			v.ymin = val
		}
		if val > v.ymax {
			v.ymax = val
		}
	}
}

// Push method appends a new value
func (v *Vector) Push(val, err float32) {
	v.Lock()
	v.data = append(v.data, val)
	v.errors = append(v.errors, err)
	if val-err < v.ymin {
		v.ymin = val - err
	}
	if val+err > v.ymax {
		v.ymax = val + err
	}
	v.Unlock()
}

// Last method returns the last entry from the vector
func (v *Vector) Last() float32 {
	return v.data[len(v.data)-1]
}

// Running mean and stddev as per http://www.johndcook.com/blog/standard_deviation/
type RunningStat struct {
	Count, Mean float64
	Var, StdDev float64
	oldM, oldV  float64
}

func (s *RunningStat) Clear() {
	s.Count = 0
	s.Mean = 0
	s.Var = 0
	s.StdDev = 0
}

func (s *RunningStat) Push(x32 float32) {
	x := float64(x32)
	s.Count++
	if s.Count == 1 {
		s.oldM, s.Mean = x, x
		s.oldV = 0
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
	return fmt.Sprintf("mean = %8.3g  std dev = %8.3g", s.Mean, s.StdDev)
}

// Buffer type is a fixed size circular buffer.
type Buffer struct {
	data []float32
	size int
}

// NewBuffer function creates a new buffer with allocated maximum size.
func NewBuffer(size int) *Buffer {
	return &Buffer{data: make([]float32, size)}
}

// Push method appends an item to the buffer.
func (b *Buffer) Push(v float32) {
	if b.size < len(b.data) {
		b.data[b.size] = v
		b.size++
	} else {
		copy(b.data, b.data[1:])
		b.data[b.size-1] = v
	}
}

// Len method returns the number of items in the buffer.
func (b *Buffer) Len() int {
	return b.size
}

// Max method returns the maximum value.
func (b *Buffer) Max() float32 {
	max := float32(-1.0e30)
	for _, v := range b.data {
		if v > max {
			max = v
		}
	}
	return max
}

// nicenum returns a "nice" number approximately equal to r
// Rounds the number if round = true. Takes the ceiling if round = false.
func Nicenum(r32 float32, round bool) float32 {
	r := float64(r32)
	exponent := math.Floor(math.Log10(r))
	fraction := r / math.Pow(10, exponent)
	var nice float64
	if round {
		if fraction < 1.5 {
			nice = 1
		} else if fraction < 3 {
			nice = 2
		} else if fraction < 7 {
			nice = 5
		} else {
			nice = 10
		}
	} else {
		if fraction <= 1 {
			nice = 1
		} else if fraction <= 2 {
			nice = 2
		} else if fraction <= 5 {
			nice = 5
		} else {
			nice = 10
		}
	}
	return float32(nice * math.Pow(10, exponent))
}

// Float32 utils
func Abs(x float32) float32 {
	if x >= 0 {
		return x
	}
	return -x
}

func Min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func Max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
