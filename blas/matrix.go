package blas

//go:generate gentype -struct native32 -type float32 native.tmpl native32.go
//go:generate gentype -struct native64 -type float64 native.tmpl native64.go

var implementation Impl

// Different implementations which can be selected
type Impl int

const (
	none Impl = iota
	Native32
	Native64
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
	Copy(m Matrix) Matrix
	Reshape(rows, cols int, shrink bool) Matrix
	Load(Ordering, ...float64) Matrix
	Data(Ordering) []float64
	Col(col1, col2 int) Matrix
	Row(row1, row2 int) Matrix
	Scale(s float64) Matrix
	Add(a, b Matrix, sc float64) Matrix
	Cmp(a, b Matrix, epsilon float64) Matrix
	Mul(a, b Matrix, aTrans, bTrans, outTrans bool) Matrix
	MulElem(a, b Matrix) Matrix
	Sum() float64
	MaxCol(m Matrix) Matrix
	Norm(m Matrix) Matrix
	SetFormat(string)
	String() string
}

// Init function is called at setup time to select the implementation
func Init(imp Impl) {
	implementation = imp
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
	case Native64:
		m = newnative64(rows, cols)
	default:
		panic("matrix implementation is not set - call init first")
	}
	return
}

// UnaryFunction interface type represents a function which can be applied elementwise to a matrix
type UnaryFunction interface {
	Apply(in, out Matrix) Matrix
}

// Unary64 represents a float64 function of one variable
type Unary64 func(float64) float64

// Apply method applies a function to each element in a matrix
func (fn Unary64) Apply(in, out Matrix) Matrix {
	switch m := out.(type) {
	case *native32:
		m.apply(in, fn)
	case *native64:
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

// Binary64 represents a float64 function of two variables
type Binary64 func(a, b float64) float64

// Apply method applies a function to each element in a matrix
func (fn Binary64) Apply(m1, m2, out Matrix) Matrix {
	switch m := out.(type) {
	case *native32:
		m.apply2(m1, m2, fn)
	case *native64:
		m.apply2(m1, m2, fn)
	default:
		panic("invalid type for apply")
	}
	return out
}
