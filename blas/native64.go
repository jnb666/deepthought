package blas

// matrix of float64 stored internally in column major order
type native64 struct {
	rows   int
	cols   int
	data   []float64
	format string
}

// constructor
func newNative64(rows, cols int) Matrix {
	return &native64{
		rows:   rows,
		cols:   cols,
		data:   make([]float64, rows*cols),
		format: "%8.4f",
	}
}

// accessors
func (m *native64) Rows() int { return m.rows }

func (m *native64) Cols() int { return m.cols }

func (m *native64) Size() int { return len(m.data) }

func (m *native64) SetFormat(f string) { m.format = f }

func (m *native64) String() string { return format(m.format, m.rows, m.cols, m.Data(RowMajor)) }

func (m *native64) at(row, col int) float64 { return m.data[m.rows*col+row] }

func (m *native64) set(row, col int, val float64) { m.data[m.rows*col+row] = val }

// Load method initialises a matrix with data from a list of float64 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *native64) Load(order Ordering, vals ...float64) Matrix {
	if len(vals) == 0 {
		panic("m64:Load - no data provided")
	}
	next := getNext(vals)
	if order == RowMajor {
		for row := 0; row < m.rows; row++ {
			for col := 0; col < m.cols; col++ {
				m.set(row, col, next())
			}
		}
	} else {
		for col := 0; col < m.cols; col++ {
			for row := 0; row < m.rows; row++ {
				m.set(row, col, next())
			}
		}
	}
	return m
}

// Dara method returns a copy of the matrix data as a slice
func (m *native64) Data(order Ordering) []float64 {
	data := make([]float64, m.rows*m.cols)
	for i := range data {
		if order == ColMajor {
			data[i] = m.data[i]
		} else {
			data[i] = m.at(i/m.cols, i%m.cols)
		}
	}
	return data
}

// Reshape method changes the dimensions without altering the data
func (m *native64) Reshape(rows, cols int) Matrix {
	if m.Size() < rows*cols {
		panic("m64:Reshape - matrix is too small")
	}
	m.rows, m.cols = rows, cols
	return m
}

// Join method joins columns from a and b and puts a copy of the result into m.
func (m *native64) Join(m1, m2 Matrix) Matrix {
	a, b := m1.(*native64), m2.(*native64)
	if a.rows != b.rows {
		panic("m64:Join - input matrices must have same number of rows")
	}
	if m.Size() < (a.cols+b.cols)*a.rows {
		panic("m64:Join - output matrix is too small")
	}
	sizea := a.rows * a.cols
	copy(m.data, a.data[:sizea])
	copy(m.data[sizea:], b.data[:b.rows*b.cols])
	m.rows, m.cols = a.rows, a.cols+b.cols
	return m
}

// Slice method returns a view on the matrix with columns [col1:col2] exclusive.
func (m *native64) Slice(col1, col2 int) Matrix {
	return &native64{
		rows:   m.rows,
		cols:   col2 - col1,
		data:   m.data[col1*m.rows : col2*m.rows],
		format: m.format,
	}
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *native64) Scale(s float64) Matrix {
	for i := range m.data[:m.rows*m.cols] {
		m.data[i] *= s
	}
	return m
}

// Add method adds a + b and puts the result in m.
func (m *native64) Add(m1, m2 Matrix) Matrix {
	checkEqualSize("m64:Add", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] + b.data[i]
	}
	return m
}

// Sub method subtracts a - b and puts the result in m.
func (m *native64) Sub(m1, m2 Matrix) Matrix {
	checkEqualSize("m64:Sub", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] - b.data[i]
	}
	return m
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
func (m *native64) MulElem(m1, m2 Matrix) Matrix {
	checkEqualSize("m64:MulElem", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] * b.data[i]
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *native64) Mul(m1, m2 Matrix, trans1, trans2 bool) Matrix {
	a, b := m1.(*native64), m2.(*native64)
	acols, arows, bcols, brows := a.cols, a.rows, b.cols, b.rows
	aix := func(row, k int) int { return row + k*arows }
	bix := func(col, k int) int { return k + col*brows }
	if trans1 {
		acols, arows = a.rows, a.cols
		aix = func(row, k int) int { return k + row*acols }
	}
	if trans2 {
		bcols, brows = b.rows, b.cols
		bix = func(col, k int) int { return col + k*bcols }
	}
	if acols != brows {
		panic("m64:Mul - mismatch in no. of rows and columns in input matrices")
	}
	m.Reshape(arows, bcols)
	for col := 0; col < m.cols; col++ {
		for row := 0; row < m.rows; row++ {
			sum := float64(0)
			for k := 0; k < acols; k++ {
				sum += a.data[aix(row, k)] * b.data[bix(col, k)]
			}
			m.set(row, col, sum)
		}
	}
	return m
}

// Unary64 represents a float64 function of one variable
type Unary64 func(float64) float64

// Apply method applies a function to each element in a matrix
func (f Unary64) Apply(in, out Matrix) Matrix {
	a, b := in.(*native64), out.(*native64)
	b.Reshape(a.rows, a.cols)
	for i, val := range a.data[:a.rows*a.cols] {
		b.data[i] = f(val)
	}
	return b
}

// Binary64 represents a float64 function of two variables
type Binary64 func(a, b float64) float64

// Apply method applies a function to each element in a matrix
func (f Binary64) Apply(m1, m2, out Matrix) Matrix {
	a, b, c := m1.(*native64), m2.(*native64), out.(*native64)
	checkEqualSize("m64:Apply", a, b, c)
	for i := range a.data[:a.rows*a.cols] {
		c.data[i] = f(a.data[i], b.data[i])
	}
	return c
}

// Sum method calculates the sum of the values in the matrix
func (m *native64) Sum() float64 {
	sum := 0.0
	for _, val := range m.data[:m.rows*m.cols] {
		sum += val
	}
	return sum
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (v *native64) MaxCol(in Matrix) Matrix {
	m := in.(*native64)
	if v.Size() < m.rows {
		panic("m64:MaxCol - output matrix is too small")
	}
	v.rows, v.cols = m.rows, 1
	for row := 0; row < m.rows; row++ {
		max, maxcol := float64(-1e38), 0
		for col := 0; col < m.cols; col++ {
			if val := m.at(row, col); val > max {
				max, maxcol = val, col
			}
		}
		v.data[row] = float64(maxcol)
	}
	return v
}
