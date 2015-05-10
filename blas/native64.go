package blas

// matrix of float64 stored internally in row major order
type native64 struct {
	rows   int
	cols   int
	stride int
	data   []float64
	format string
}

// constructor
func newnative64(rows, cols int) Matrix {
	return &native64{
		rows:   rows,
		cols:   cols,
		stride: cols,
		data:   make([]float64, rows*cols),
		format: "%8.4f",
	}
}

func (m *native64) Release() {}

// accessors
func (m *native64) Rows() int { return m.rows }

func (m *native64) Cols() int { return m.cols }

func (m *native64) Size() int { return len(m.data) }

func (m *native64) SetFormat(f string) { m.format = f }

func (m *native64) String() string { 
	return format(m.format, m.rows, m.cols, m.Data(RowMajor))
}

func (m *native64) at(row, col int) float64 {
	return m.data[row*m.stride + col] 
}

func (m *native64) set(row, col int, val float64) { 
	m.data[row*m.stride + col] = val
}

// Load method initialises a matrix with data from a list of float64 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *native64) Load(order Ordering, vals ...float64) Matrix {
	if len(vals) == 0 {
		panic("blas:Load - no data provided")
	}
	next := getNext(vals)
	if order == RowMajor {
		for row := 0; row < m.rows; row++ {
			for col := 0; col < m.cols; col++ {
				m.set(row, col, float64(next()))
			}
		}
	} else {
		for col := 0; col < m.cols; col++ {
			for row := 0; row < m.rows; row++ {
				m.set(row, col, float64(next()))
			}
		}
	}
	return m
}

// Data method returns a copy of the matrix data as a slice
func (m *native64) Data(order Ordering) []float64 {
	data := make([]float64, m.rows*m.cols)
	i := 0
	if order == RowMajor {
		for row := 0; row < m.rows; row++ {
			for col := 0; col < m.cols; col++ {
				data[i] = float64(m.at(row, col))
				i++
			}
		}
	} else {
		for col := 0; col < m.cols; col++ {
			for row := 0; row < m.rows; row++ {
				data[i] = float64(m.at(row, col))
				i++
			}
		}
	}
	return data
}

// Copy method returns a copy of the input matrix
func (m *native64) Copy(in Matrix) Matrix {
	a := in.(*native64)
	m.Reshape(a.rows, a.cols, false)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, a.at(row, col))
		}
	}
	return m
}

// Reshape method changes the dimensions without altering the data
func (m *native64) Reshape(rows, cols int, shrink bool) Matrix {
	if m.Size() < rows*cols {
		panic("blas:Reshape - matrix is too small")
	}
	m.rows, m.cols = rows, cols
	if shrink || m.stride < m.cols {
		m.stride = m.cols
	}
	return m
}

// Row method returns a view on the matrix with rows [r1:r2] exclusive.
func (m *native64) Row(r1, r2 int) Matrix {
	return &native64{
		rows:	r2-r1,
		cols:	m.cols,
		stride: m.stride,
		data:	m.data[r1*m.stride:],
		format: m.format,
	}
}

// Col method returns a view on the matrix with columns [c1:c2] exclusive.
func (m *native64) Col(c1, c2 int) Matrix {
	return &native64{
		rows:	m.rows,
		cols:	c2-c1,
		stride: m.stride,
		data:	m.data[c1:],
		format: m.format,
	}
}

// Set method sets all elements of the matrix to the given value
func (m *native64) Set(val float64) Matrix {
	value := float64(val)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, value)
		}
	}
	return m
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *native64) Scale(s float64) Matrix {
	sc := float64(s)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, m.at(row, col)*sc)
		}
	}
	return m
}

// Add method evaluates a + sc * b and puts the result in m.
func (m *native64) Add(m1, m2 Matrix, sc float64) Matrix {
	checkEqualSize("blas:Add", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	scale := float64(sc)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, a.at(row, col) + scale*b.at(row, col))
		}
	}
	return m
}

// Cmp method compares a and b and returns a matrix with 0 where they are equal else 1.
func (m *native64) Cmp(m1, m2 Matrix, epsilon float64) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	eps := float64(epsilon)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			ax, bx := a.at(row, col), b.at(row, col)
			if ax < bx-eps || ax > bx+eps {
				m.set(row, col, 1) 
			} else {
				m.set(row, col, 0) 
			}
		}
	}
	return m
}

func transnative64(a, b *native64, atrans, btrans bool) (ac, ar, bc, br int, aix, bix func(r, c int)int) {
	ac, ar, bc, br = a.cols, a.rows, b.cols, b.rows
	aix = func(r, c int) int { return r*a.stride + c }
	bix = func(r, c int) int { return r*b.stride + c }
	if atrans {
		ac, ar = a.rows, a.cols
		aix = func(r, c int) int { return c*a.stride + r }
	}
	if btrans {
		bc, br = b.rows, b.cols
		bix = func(r, c int) int { return c*b.stride + r }
	}
	return
}

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *native64) MulElem(m1, m2 Matrix, trans1, trans2 bool) Matrix {
	a, b := m1.(*native64), m2.(*native64)
	acols, arows, bcols, brows, aix, bix := transnative64(a, b, trans1, trans2)
	if acols != bcols || arows != brows {
		panic("blas:MulElem - mismatch in no. of rows and columns in input matrices")
	}
	m.Reshape(arows, acols, true)
	for row := 0; row < arows; row++ {
		for col := 0; col < acols; col++ {
			m.set(row, col, a.data[aix(row, col)] * b.data[bix(row, col)])
		}
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *native64) Mul(m1, m2 Matrix, trans1, trans2 bool) Matrix {
	a, b := m1.(*native64), m2.(*native64)
	acols, arows, bcols, brows, aix, bix := transnative64(a, b, trans1, trans2)
	if acols != brows {
		panic("blas:Mul - mismatch in no. of rows and columns in input matrices")
	}
	m.Reshape(arows, bcols, true)
	for row := 0; row < arows; row++ {
		for col := 0; col < bcols; col++ {
			sum := float64(0)
			for k := 0; k < acols; k++ {
				sum += a.data[aix(row, k)] * b.data[bix(k, col)]
			}
			m.set(row, col, sum)
		}
	}
	return m
}

// Sum method calculates the sum of the values in the matrix
func (m *native64) Sum() float64 {
	sum := 0.0
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			sum += float64(m.at(row, col))
		}
	}
	return sum
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (v *native64) MaxCol(in Matrix) Matrix {
	m := in.(*native64)
	if v.Size() < m.rows {
		panic("blas:MaxCol - output matrix is too small")
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

// Norm method divides each element by the sum of the values in that row.
func (m *native64) Norm(in Matrix) Matrix {
	a := in.(*native64)
	m.Reshape(a.rows, a.cols, false)
	for row := 0; row < m.rows; row++ {
		sum := 0.0
		for col := 0; col < m.cols; col++ {
			sum += float64(a.at(row, col))
		}
		scale := 1.0 / float64(sum)
		for col := 0; col < m.cols; col++ {
			m.set(row, col, scale * a.at(row, col))
		}
	}
	return m
}

func (m *native64) apply(in Matrix, fn Unary64) {
	a := in.(*native64)
	m.Reshape(a.rows, a.cols, false)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			val := float64(a.at(row, col))
			m.set(row, col, float64(fn(val)))
		}
	}
}

func (m *native64) apply2(m1, m2 Matrix, fn Binary64) {
	checkEqualSize("binary64:Apply", m1, m2, m)
	a, b := m1.(*native64), m2.(*native64)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			v1 := float64(a.at(row, col))
			v2 := float64(b.at(row, col))
			m.set(row, col, float64(fn(v1, v2)))
		}
	}
}