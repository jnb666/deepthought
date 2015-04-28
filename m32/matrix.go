package m32

import (
	"fmt"
	"math/rand"
	"strings"
)

// Switch for column major or row major data represntation
type Ordering bool

const (
	RowMajor Ordering = false
	ColMajor Ordering = true
)

// Matrix is a struct type representing a dense matrix of float32 numbers.
// Data is stored internally in row major order.
type Matrix struct {
	Rows   int
	Cols   int
	data   []float32
	format string
}

// New function creates a new matrix of given size.
func New(rows, cols int) *Matrix {
	return &Matrix{Rows: rows, Cols: cols, data: make([]float32, rows*cols), format: "%8.4g"}
}

// Copy method copies data from matrix a into m
func (m *Matrix) Copy(a *Matrix) *Matrix {
	if len(m.data) < a.Rows*a.Cols {
		panic("m32:Copy - output matrix is too small")
	}
	m.Rows, m.Cols = a.Rows, a.Cols
	copy(m.data, a.data[:a.Rows*a.Cols])
	return m
}

// Transpose method updates the data in place to transpose the matrix.
func (m *Matrix) Transpose() *Matrix {
	temp := make([]float32, m.Rows*m.Cols)
	copy(temp, m.data[:m.Rows*m.Cols])
	m.Rows, m.Cols = m.Cols, m.Rows
	for row := 0; row < m.Rows; row++ {
		for col := 0; col < m.Cols; col++ {
			m.data[row*m.Cols+col] = temp[col*m.Rows+row]
		}
	}
	return m
}

// Slice method returns a view on the matrix containing rows from start to end exclusive.
func (m *Matrix) Slice(start, end int) *Matrix {
	if end < start || start < 0 || end > m.Rows {
		panic("m32:Slice - row parameters out of range")
	}
	return &Matrix{
		Rows:   end - start,
		Cols:   m.Cols,
		data:   m.data[start*m.Cols : end*m.Cols],
		format: m.format,
	}
}

// Load method initialises a matrix with data from a list of float32 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *Matrix) Load(order Ordering, vals ...float32) *Matrix {
	if len(vals) == 0 {
		panic("m32:Load - no data provided")
	}
	j := 0
	next := func() (v float32) {
		v = vals[j]
		j++
		if j == len(vals) {
			j = 0
		}
		return
	}
	if order == RowMajor {
		// data is provided in same ordering as we store it internally
		for row := 0; row < m.Rows; row++ {
			for col := 0; col < m.Cols; col++ {
				m.data[row*m.Cols+col] = next()
			}
		}
	} else {
		// data is transposed
		for col := 0; col < m.Cols; col++ {
			for row := 0; row < m.Rows; row++ {
				m.data[row*m.Cols+col] = next()
			}
		}
	}
	return m
}

// SetFormat method sets the printf format string used by the String method for each matrix element.
func (m *Matrix) SetFormat(s string) {
	m.format = s
}

// String method formats a matrix for printing.
func (m *Matrix) String() string {
	if m.Rows == 0 {
		return "[]"
	}
	str := make([]string, m.Rows)
	for row := 0; row < m.Rows; row++ {
		var begin, end string
		switch {
		case m.Rows == 1:
			begin, end = "[", "]"
		case row == 0:
			begin, end = "⎡", "⎤"
		case row == m.Rows-1:
			begin, end = "⎣", "⎦"
		default:
			begin, end = "⎢", "⎥"
		}
		str[row] = begin
		for col := 0; col < m.Cols; col++ {
			str[row] += fmt.Sprintf(" "+m.format, m.data[row*m.Cols+col])
		}
		str[row] += " " + end
	}
	return strings.Join(str, "\n")
}

// Random method sets the matrix elements to uniform random numbers.
func (m *Matrix) Random(min, max float32) *Matrix {
	for i := range m.data[:m.Rows*m.Cols] {
		m.data[i] = min + (max-min)*rand.Float32()
	}
	return m
}

// Apply method updates each element of a matrix using the given function.
func (m *Matrix) Apply(fn func(float32) float32) *Matrix {
	for i := range m.data[:m.Rows*m.Cols] {
		m.data[i] = fn(m.data[i])
	}
	return m
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *Matrix) Scale(s float32) *Matrix {
	for i := range m.data[:m.Rows*m.Cols] {
		m.data[i] *= s
	}
	return m
}

// Add method evaluates s*a + b and puts the result in m.
func (m *Matrix) Add(s float32, a, b *Matrix) *Matrix {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		panic("m32:Add - mismatch in no. of rows and columns in input matrices")
	}
	if len(m.data) < a.Rows*a.Cols {
		panic("m32:Add - output matrix is too small")
	}
	m.Rows, m.Cols = a.Rows, a.Cols
	for i := range m.data[:m.Rows*m.Cols] {
		m.data[i] = s*a.data[i] + b.data[i]
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// If the aTrans flag is set then transpose matrix a.
func (m *Matrix) Mul(a, b *Matrix, aTrans bool) *Matrix {
	if (aTrans && a.Rows != b.Rows) || (!aTrans && a.Cols != b.Rows) {
		panic("m32:Mul - mismatch in no. of rows and columns in input matrices")
	}
	if (aTrans && len(m.data) < a.Cols*b.Cols) || (!aTrans && len(m.data) < a.Rows*b.Cols) {
		panic("m32:Mul - output matrix is too small")
	}
	if aTrans {
		m.Rows, m.Cols = a.Cols, b.Cols
		for row := 0; row < m.Rows; row++ {
			for col := 0; col < m.Cols; col++ {
				sum := float32(0)
				for k := 0; k < a.Rows; k++ {
					sum += a.data[k*a.Cols+row] * b.data[k*b.Cols+col]
				}
				m.data[row*m.Cols+col] = sum
			}
		}
	} else {
		m.Rows, m.Cols = a.Rows, b.Cols
		for row := 0; row < m.Rows; row++ {
			for col := 0; col < m.Cols; col++ {
				sum := float32(0)
				for k := 0; k < a.Cols; k++ {
					sum += a.data[row*a.Cols+k] * b.data[k*b.Cols+col]
				}
				m.data[row*m.Cols+col] = sum
			}
		}
	}
	return m
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
func (m *Matrix) MulElem(a, b *Matrix) *Matrix {
	if a.Cols != b.Cols || a.Rows != b.Rows {
		panic("m32:MulElem - mismatch in no. of rows and columns in input matrices")
	}
	if len(m.data) < a.Rows*a.Cols {
		panic("m32:MulElem - output matrix is too small")
	}
	m.Rows, m.Cols = a.Rows, a.Cols
	for i := range m.data[:m.Rows*m.Cols] {
		m.data[i] = a.data[i] * b.data[i]
	}
	return m
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (v *Matrix) MaxCol(m *Matrix) *Matrix {
	if len(v.data) < m.Rows {
		panic("m32:MaxCol - output matrix is too small")
	}
	v.Rows, v.Cols = m.Rows, 1
	for row := 0; row < m.Rows; row++ {
		max, maxcol := float32(-1e38), 0
		for col := 0; col < m.Cols; col++ {
			if m.data[row*m.Cols+col] > max {
				max, maxcol = m.data[row*m.Cols+col], col
			}
		}
		v.data[row] = float32(maxcol)
	}
	return v
}

// SumDiff2 function calculates the sum of the elementwise differences between two matrices.
func SumDiff2(a, b *Matrix) float32 {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("m32:SumDiff2 - matrix size mismatch")
	}
	sum := float32(0)
	for i := range a.data[:a.Rows*a.Cols] {
		diff := a.data[i] - b.data[i]
		sum += diff * diff
	}
	return sum
}

// CountDiff2 function calculates the number of elements which differ between two matrices by more than epsilon.
func CountDiff(a, b *Matrix, epsilon float32) float32 {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("m32:CountDiff - matrix size mismatch")
	}
	count := 0
	for i := range a.data[:a.Rows*a.Cols] {
		diff := a.data[i] - b.data[i]
		if diff < -epsilon || diff > epsilon {
			count++
		}
	}
	return float32(count)
}
