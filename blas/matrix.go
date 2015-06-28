// Package blas contains linear algebra routines for matrix manipulation with optional OpenCL acceleration.
package blas

import (
	"math/rand"
	"time"
)

var implementation Impl

// Different implementations which can be selected
type Impl int

const (
	none Impl = iota
	Native32
	OpenCL32
)

// Switch for column major or row major data represntation
type Ordering bool

const (
	RowMajor Ordering = false
	ColMajor Ordering = true
)

// Matrix interface type represents a fix size matrix of row x cols.
type Matrix interface {
	Rows() int
	Cols() int
	Size() int
	Release()
	Copy(m, ix Matrix) Matrix
	Transpose(m Matrix) Matrix
	Reshape(rows, cols int, shrink bool) Matrix
	Set(val float32) Matrix
	Load(Ordering, ...float32) Matrix
	Random(min, max float32) Matrix
	Data(Ordering) []float32
	Col(col1, col2 int) Matrix
	Row(row1, row2 int) Matrix
	Scale(s float32) Matrix
	Add(a, b Matrix, sc float32) Matrix
	Cmp(a, b Matrix, epsilon float32) Matrix
	Mul(a, b Matrix, aTrans, bTrans, oTrans bool) Matrix
	MulElem(a, b Matrix) Matrix
	Sum() float32
	SumRows(a Matrix) Matrix
	MaxCol(m Matrix) Matrix
	Norm(m Matrix) Matrix
	Histogram(m Matrix, bins int, min, max float32) Matrix
	SetFormat(string)
	String() string
}

// Init function is called at setup time to select the implementation
func Init(imp Impl) {
	implementation = imp
	if implementation == OpenCL32 {
		initCL()
	}
}

// SeedRandom function sets the random seed, or seeds using time if input is zero. Returns the seed which was used.
func SeedRandom(seed int64) int64 {
	if seed == 0 {
		seed = time.Now().UTC().UnixNano()
	}
	rand.Seed(seed)
	if implementation == OpenCL32 {
		initSeeds(rand.Int31())
	}
	return seed
}

// Implementation function returns the current implementation
func Implementation() Impl {
	return implementation
}

// New function creates a new matrix of the given size
func New(rows, cols int) (m Matrix) {
	switch implementation {
	case Native32:
		m = newnative32(rows, cols)
	case OpenCL32:
		m = newopencl32(rows, cols)
	default:
		panic("matrix implementation is not set - call init first")
	}
	return
}

// Release function is called at shutdown to release any resources
func Release() {
	if implementation == OpenCL32 {
		releaseCL()
	}
}

// UnaryFunction interface type represents a function which can be applied elementwise to a matrix
type UnaryFunction interface {
	Apply(in, out Matrix) Matrix
}

// Unary32 represents a float32 function of one variable
type Unary32 func(float32) float32

// Apply method applies a function to each element in a matrix
func (fn Unary32) Apply(in, out Matrix) Matrix {
	switch m := out.(type) {
	case *native32:
		m.apply(in, fn)
	default:
		panic("invalid type for apply")
	}
	return out
}

// Function2 interface applies a function elementwise to two matrices
type BinaryFunction interface {
	Apply(a, b, out Matrix) Matrix
}

// Binary32 represents a float32 function of two variables
type Binary32 func(a, b float32) float32

// Apply method applies a function to each element in a matrix
func (fn Binary32) Apply(m1, m2, out Matrix) Matrix {
	switch m := out.(type) {
	case *native32:
		m.apply2(m1, m2, fn)
	default:
		panic("invalid type for apply")
	}
	return out
}
