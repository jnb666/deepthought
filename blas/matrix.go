package blas

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
	Reshape(rows, cols int) Matrix
	Load(Ordering, ...float64) Matrix
	Data(Ordering) []float64
	Join(a, b Matrix) Matrix
	Slice(col1, col2 int) Matrix
	Row(i int) Matrix
	Scale(s float64) Matrix
	Add(a, b Matrix) Matrix
	Sub(a, b Matrix) Matrix
	Cmp(a, b Matrix, epsilon float64) Matrix
	Mul(a, b Matrix, aTrans, bTrans, outTrans bool) Matrix
	MulElem(a, b Matrix) Matrix
	Sum() float64
	MaxCol(m Matrix) Matrix
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
		m = newNative32(rows, cols)
	case Native64:
		m = newNative64(rows, cols)
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
	out.Reshape(in.Rows(), in.Cols())
	switch implementation {
	case Native32:
		a, b := in.(*native32), out.(*native32)
		for i, val := range a.data[:a.rows*a.cols] {
			b.data[i] = float32(fn(float64(val)))
		}
	case Native64:
		a, b := in.(*native64), out.(*native64)
		for i, val := range a.data[:a.rows*a.cols] {
			b.data[i] = fn(val)
		}
	default:
		panic("invalid implementation for apply")
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
	checkEqualSize("binary64:Apply", m1, m2, out)
	switch implementation {
	case Native32:
		a, b, c := m1.(*native32), m2.(*native32), out.(*native32)
		for i := range a.data[:a.rows*a.cols] {
			c.data[i] = float32(fn((float64(a.data[i])), float64(b.data[i])))
		}
	case Native64:
		a, b, c := m1.(*native64), m2.(*native64), out.(*native64)
		for i := range a.data[:a.rows*a.cols] {
			c.data[i] = fn(a.data[i], b.data[i])
		}
	default:
		panic("invalid implementation for apply")
	}
	return out
}
