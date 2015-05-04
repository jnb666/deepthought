package blas

// matrix of float32 stored internally in column major order
type native32 struct {
	rows   int
	cols   int
	data   []float32
	format string
}

// constructor
func newNative32(rows, cols int) Matrix {
	return &native32{
		rows:   rows,
		cols:   cols,
		data:   make([]float32, rows*cols),
		format: "%8.4f",
	}
}

// accessors
func (m *native32) Rows() int { return m.rows }

func (m *native32) Cols() int { return m.cols }

func (m *native32) Size() int { return len(m.data) }

func (m *native32) SetFormat(f string) { m.format = f }

func (m *native32) String() string { return format(m.format, m.rows, m.cols, m.Data(RowMajor)) }

func (m *native32) at(row, col int) float32 { return m.data[m.rows*col+row] }

func (m *native32) set(row, col int, val float32) { m.data[m.rows*col+row] = val }

// Load method initialises a matrix with data from a list of float32 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *native32) Load(order Ordering, vals ...float64) Matrix {
	if len(vals) == 0 {
		panic("m32:Load - no data provided")
	}
	next := getNext(vals)
	if order == RowMajor {
		for row := 0; row < m.rows; row++ {
			for col := 0; col < m.cols; col++ {
				m.set(row, col, float32(next()))
			}
		}
	} else {
		for col := 0; col < m.cols; col++ {
			for row := 0; row < m.rows; row++ {
				m.set(row, col, float32(next()))
			}
		}
	}
	return m
}

// Dara method returns a copy of the matrix data as a slice
func (m *native32) Data(order Ordering) []float64 {
	data := make([]float64, m.rows*m.cols)
	for i := range data {
		if order == ColMajor {
			data[i] = float64(m.data[i])
		} else {
			data[i] = float64(m.at(i/m.cols, i%m.cols))
		}
	}
	return data
}

// Reshape method changes the dimensions without altering the data
func (m *native32) Reshape(rows, cols int) Matrix {
	if m.Size() < rows*cols {
		panic("m32:Reshape - matrix is too small")
	}
	m.rows, m.cols = rows, cols
	return m
}

// Join method joins columns from a and b and puts a copy of the result into m.
func (m *native32) Join(m1, m2 Matrix) Matrix {
	a, b := m1.(*native32), m2.(*native32)
	if a.rows != b.rows {
		panic("m32:Join - input matrices must have same number of rows")
	}
	if m.Size() < (a.cols+b.cols)*a.rows {
		panic("m32:Join - output matrix is too small")
	}
	sizea := a.rows * a.cols
	copy(m.data, a.data[:sizea])
	copy(m.data[sizea:], b.data[:b.rows*b.cols])
	m.rows, m.cols = a.rows, a.cols+b.cols
	return m
}

// Slice method returns a view on the matrix with columns [col1:col2] exclusive.
func (m *native32) Slice(col1, col2 int) Matrix {
	return &native32{
		rows:   m.rows,
		cols:   col2 - col1,
		data:   m.data[col1*m.rows : col2*m.rows],
		format: m.format,
	}
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *native32) Scale(s float64) Matrix {
	sc := float32(s)
	for i := range m.data[:m.rows*m.cols] {
		m.data[i] *= sc
	}
	return m
}

// Add method adds a + b and puts the result in m.
func (m *native32) Add(m1, m2 Matrix) Matrix {
	checkEqualSize("m32:Add", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] + b.data[i]
	}
	return m
}

// Sub method subtracts a - b and puts the result in m.
func (m *native32) Sub(m1, m2 Matrix) Matrix {
	checkEqualSize("m32:Sub", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] - b.data[i]
	}
	return m
}

// Cmp method compares a and b and returns a matrix with 0 where they are equal else 1.
func (m *native32) Cmp(m1, m2 Matrix, epsilon float64) Matrix {
	checkEqualSize("m32:Cmp", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	eps := float32(epsilon)
	for i := range m.data[:a.rows*a.cols] {
		if a.data[i] < b.data[i]-eps || a.data[i] > b.data[i]+eps {
			m.data[i] = 1
		} else {
			m.data[i] = 0
		}
	}
	return m
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
func (m *native32) MulElem(m1, m2 Matrix) Matrix {
	checkEqualSize("m32:MulElem", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	for i := range m.data[:a.rows*a.cols] {
		m.data[i] = a.data[i] * b.data[i]
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *native32) Mul(m1, m2 Matrix, trans1, trans2, outTrans bool) Matrix {
	a, b := m1.(*native32), m2.(*native32)
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
		panic("m32:Mul - mismatch in no. of rows and columns in input matrices")
	}
	m.Reshape(arows, bcols)
	for col := 0; col < m.cols; col++ {
		for row := 0; row < m.rows; row++ {
			sum := float32(0)
			for k := 0; k < acols; k++ {
				sum += a.data[aix(row, k)] * b.data[bix(col, k)]
			}
			if outTrans {
				m.data[m.cols*row+col] = sum
			} else {
				m.data[m.rows*col+row] = sum
			}
		}
	}
	if outTrans {
		m.rows, m.cols = m.cols, m.rows
	}
	return m
}

// Unary32 represents a float32 function of one variable
type Unary32 func(float32) float32

// Apply method applies a function to each element in a matrix
func (f Unary32) Apply(in, out Matrix) Matrix {
	a, b := in.(*native32), out.(*native32)
	b.Reshape(a.rows, a.cols)
	for i, val := range a.data[:a.rows*a.cols] {
		b.data[i] = f(val)
	}
	return b
}

// Binary32 represents a float32 function of two variables
type Binary32 func(a, b float32) float32

// Apply method applies a function to each element in a matrix
func (f Binary32) Apply(m1, m2, out Matrix) Matrix {
	a, b, c := m1.(*native32), m2.(*native32), out.(*native32)
	checkEqualSize("m32:Apply", a, b, c)
	for i := range a.data[:a.rows*a.cols] {
		c.data[i] = f(a.data[i], b.data[i])
	}
	return c
}

// Sum method calculates the sum of the values in the matrix
func (m *native32) Sum() float64 {
	sum := 0.0
	for _, val := range m.data[:m.rows*m.cols] {
		sum += float64(val)
	}
	return sum
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (v *native32) MaxCol(in Matrix) Matrix {
	m := in.(*native32)
	if v.Size() < m.rows {
		panic("m32:MaxCol - output matrix is too small")
	}
	v.rows, v.cols = m.rows, 1
	for row := 0; row < m.rows; row++ {
		max, maxcol := float32(-1e38), 0
		for col := 0; col < m.cols; col++ {
			if val := m.at(row, col); val > max {
				max, maxcol = val, col
			}
		}
		v.data[row] = float32(maxcol)
	}
	return v
}
