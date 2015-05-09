package blas

import (
	"fmt"
	"math"
	"strings"
)

// iterator to read from a circular buffer
func getNext(vals []float64) func() float64 {
	j := 0
	return func() float64 {
		v := vals[j]
		j++
		if j == len(vals) {
			j = 0
		}
		return v
	}
}

// format a matrix for printing. %c is compact single char format
func format(f string, rows, cols int, data []float64) string {
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

func toChar(v float64) byte {
	if v > 1e-4 {
		return 'A' + byte(math.Min(v, 1)*25)
	}
	if v < -1e-4 {
		return 'a' + byte(math.Min(-v, 1)*25)
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
