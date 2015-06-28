package blas

import (
	"fmt"
	"strings"
)

// iterator to read from a circular buffer
func getNext(vals []float32) func() float32 {
	j := 0
	return func() float32 {
		v := vals[j]
		j++
		if j == len(vals) {
			j = 0
		}
		return v
	}
}

// format a matrix for printing. %c is compact single char format
func format(f string, rows, cols int, data []float32) string {
	if rows == 0 {
		return "[]"
	}
	str := make([]string, rows)
	for row := 0; row < rows; row++ {
		var begin, end string
		switch {
		case rows == 1:
			begin, end = "[", "]"
		case row == 0:
			begin, end = "⎡", "⎤"
		case row == rows-1:
			begin, end = "⎣", "⎦"
		default:
			begin, end = "⎢", "⎥"
		}
		str[row] = begin
		for col := 0; col < cols; col++ {
			val := data[cols*row+col]
			if f == "%c" {
				str[row] += string(toChar(val))
			} else {
				str[row] += fmt.Sprintf(" "+f, val)
			}
		}
		if f != "%c" {
			str[row] += "  "
		}
		str[row] += end
	}
	return strings.Join(str, "\n")
}

func toChar(v float32) byte {
	if v > 1e-4 {
		return 'A' + byte(fmin(v, 1)*25)
	}
	if v < -1e-4 {
		return 'a' + byte(fmin(-v, 1)*25)
	}
	return '.'
}

// common sanity checks
func checkEqualSize(caller string, a, b, out Matrix) {
	rows, cols := a.Rows(), a.Cols()
	if cols != b.Cols() || rows != b.Rows() {
		panic(caller + " - mismatch in no. of rows and columns in input matrices")
	}
	out.Reshape(rows, cols, false)
}

func fmin(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// functions for modular arithmetic - used by random seeds
func addMod64(a, b, M uint64) (v uint64) {
	v = a + b
	if (v >= M) || (v < a) {
		v -= M
	}
	return
}

func mulMod64(a, b, M uint64) (r uint64) {
	for a != 0 {
		if a&1 != 0 {
			r = addMod64(r, b, M)
		}
		b = addMod64(b, b, M)
		a = a >> 1
	}
	return
}

func powMod64(a, e, M uint64) (acc uint64) {
	sqr := a
	acc = 1
	for e != 0 {
		if e&1 != 0 {
			acc = mulMod64(acc, sqr, M)
		}
		sqr = mulMod64(sqr, sqr, M)
		e = e >> 1
	}
	return
}
