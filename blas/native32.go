package blas

// matrix of float32 stored internally in row major order
type native32 struct {
	rows   int
	cols   int
	stride int
	data   []float32
	format string
}

// constructor
func newnative32(rows, cols int) Matrix {
	return &native32{
		rows:   rows,
		cols:   cols,
		stride: cols,
		data:   make([]float32, rows*cols),
		format: "%8.4f",
	}
}

func (m *native32) Release() {}

// accessors
func (m *native32) Rows() int { return m.rows }

func (m *native32) Cols() int { return m.cols }

func (m *native32) Size() int { return len(m.data) }

func (m *native32) SetFormat(f string) { m.format = f }

func (m *native32) String() string { 
	return format(m.format, m.rows, m.cols, m.Data(RowMajor))
}

func (m *native32) at(row, col int) float32 {
	return m.data[row*m.stride + col] 
}

func (m *native32) set(row, col int, val float32) { 
	m.data[row*m.stride + col] = val
}

// Load method initialises a matrix with data from a list of float64 values.
// If the number of values is less than the size then they are repeated to fill the matrix.
func (m *native32) Load(order Ordering, vals ...float64) Matrix {
	if len(vals) == 0 {
		panic("blas:Load - no data provided")
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

// Data method returns a copy of the matrix data as a slice
func (m *native32) Data(order Ordering) []float64 {
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
// if ix matrix is non-nil then this provides a column vector with the rows to copy.
func (m *native32) Copy(in, ix Matrix) Matrix {
	a := in.(*native32)
	if ix == nil {
		m.Reshape(a.rows, a.cols, false)
		for row := 0; row < m.rows; row++ {
			for col := 0; col < m.cols; col++ {
				m.set(row, col, a.at(row, col))
			}
		}
	} else {
		b := ix.(*native32)
		m.Reshape(b.rows, a.cols, false)
		for row := 0; row < b.rows; row++ {
			ixrow := int(b.at(row, 0))
			for col := 0; col < m.cols; col++ {
				m.set(row, col, a.at(ixrow, col))
			}
		}		
	}
	return m
}

// Transpose method returns a transposed copy of the input matrix
func (m *native32) Transpose(in Matrix) Matrix {
	a := in.(*native32)
	m.Reshape(a.cols, a.rows, true)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, a.at(col, row))
		}
	}
	return m
}

// Reshape method changes the dimensions without altering the data
func (m *native32) Reshape(rows, cols int, shrink bool) Matrix {
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
func (m *native32) Row(r1, r2 int) Matrix {
	return &native32{
		rows:	r2-r1,
		cols:	m.cols,
		stride: m.stride,
		data:	m.data[r1*m.stride:],
		format: m.format,
	}
}

// Col method returns a view on the matrix with columns [c1:c2] exclusive.
func (m *native32) Col(c1, c2 int) Matrix {
	return &native32{
		rows:	m.rows,
		cols:	c2-c1,
		stride: m.stride,
		data:	m.data[c1:],
		format: m.format,
	}
}

// Set method sets all elements of the matrix to the given value
func (m *native32) Set(val float64) Matrix {
	value := float32(val)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, value)
		}
	}
	return m
}

// Scale method muliplies each element of the matrix by a scalar.
func (m *native32) Scale(s float64) Matrix {
	sc := float32(s)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, m.at(row, col)*sc)
		}
	}
	return m
}

// Add method evaluates a + sc * b and puts the result in m.
func (m *native32) Add(m1, m2 Matrix, sc float64) Matrix {
	checkEqualSize("blas:Add", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	scale := float32(sc)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			m.set(row, col, a.at(row, col) + scale*b.at(row, col))
		}
	}
	return m
}

// Cmp method compares a and b and returns a matrix with 0 where they are equal else 1.
func (m *native32) Cmp(m1, m2 Matrix, epsilon float64) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	eps := float32(epsilon)
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

// MulElem method performs element wise multiplication of the two input matrices and puts the output in m.
// If trans1, trans2 flag is set then input matrices are transposed first.
func (m *native32) MulElem(m1, m2 Matrix) Matrix {
	checkEqualSize("blas:Cmp", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	for row := 0; row < a.rows; row++ {
		for col := 0; col < a.cols; col++ {
			m.set(row, col, a.at(row, col) * b.at(row, col))
		}
	}
	return m
}

// Mul method multiplies two matrices using regular matrix multiplication and puts the output in m.
func (m *native32) Mul(m1, m2 Matrix, aTrans, bTrans, oTrans bool) Matrix {
	a, b := m1.(*native32), m2.(*native32)
	ar, ac, br, bc := a.rows, a.cols, b.rows, b.cols
	aget := func(r, c int) float32 { return a.at(r, c)} 
	bget := func(r, c int) float32 { return b.at(r, c)}

	if aTrans {
		ar, ac = ac, ar
		aget = func(r, c int) float32 { return a.at(c, r)} 
	}
	if bTrans {
		br, bc = bc, br
		bget = func(r, c int) float32 { return b.at(c, r)}
	}
	if ac != br {
		panic("blas:Mul - mismatch in no. of rows and columns in input matrices")
	}
	if oTrans {
		m.Reshape(bc, ar, true)
		for row := 0; row < ar; row++ {
			for col := 0; col < bc; col++ {
				sum := float32(0)
				for k := 0; k < ac; k++ {
					sum += aget(row, k) * bget(k, col)
				}
				m.set(col, row, sum)
			}
		}
	} else {
		m.Reshape(ar, bc, true)
		for row := 0; row < ar; row++ {
			for col := 0; col < bc; col++ {
				sum := float32(0)
				for k := 0; k < ac; k++ {
					sum += aget(row, k) * bget(k, col)
				}
				m.set(row, col, sum)
			}
		}
	}
	return m
}

// Sum method calculates the sum of the values in the matrix
func (m *native32) Sum() float64 {
	sum := 0.0
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			sum += float64(m.at(row, col))
		}
	}
	return sum
}

// SumRows method returns a column vector with the sum of each row.
func (m *native32) SumRows(in Matrix) Matrix {
	a := in.(*native32)
	m.Reshape(a.rows, 1, false)	
	for row := 0; row < a.rows; row++ {
		sum := float32(0)
		for col := 0; col < a.cols; col++ {
			sum += a.at(row, col)
		}
		m.set(row, 0, sum)
	}
	return m
}

// MaxCol method gets the column number with the maximim value for each row of the input matrix.
func (v *native32) MaxCol(in Matrix) Matrix {
	m := in.(*native32)
	if v.Size() < m.rows {
		panic("blas:MaxCol - output matrix is too small")
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

// Norm method divides each element by the sum of the values in that row.
func (m *native32) Norm(in Matrix) Matrix {
	a := in.(*native32)
	m.Reshape(a.rows, a.cols, false)
	for row := 0; row < m.rows; row++ {
		sum := 0.0
		for col := 0; col < m.cols; col++ {
			sum += float64(a.at(row, col))
		}
		scale := 1.0 / float32(sum)
		for col := 0; col < m.cols; col++ {
			m.set(row, col, scale * a.at(row, col))
		}
	}
	return m
}

// Histogram method adds bins the values from the input column vector and adds to the histogram.
func (m *native32) Histogram(in Matrix, bins int, min, max float64) Matrix {
	a := in.(*native32)
	m.Reshape(bins, 1, false)
	scale := float32(bins) / float32(max-min)
	xmin := float32(min)
	for row := 0; row < a.rows; row++ {
		bin := int(scale * (a.at(row, 0) - xmin))
		if bin < 0 {
			bin = 0
		}
		if bin >= bins {
			bin = bins-1
		}
		m.data[bin*m.stride]++
	}
	return m
}

func (m *native32) apply(in Matrix, fn Unary64) {
	a := in.(*native32)
	m.Reshape(a.rows, a.cols, false)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			val := float64(a.at(row, col))
			m.set(row, col, float32(fn(val)))
		}
	}
}

func (m *native32) apply2(m1, m2 Matrix, fn Binary64) {
	checkEqualSize("binary64:Apply", m1, m2, m)
	a, b := m1.(*native32), m2.(*native32)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			v1 := float64(a.at(row, col))
			v2 := float64(b.at(row, col))
			m.set(row, col, float32(fn(v1, v2)))
		}
	}
}