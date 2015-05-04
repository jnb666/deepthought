package blas

import (
	"fmt"
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

// format a matrix for printing.
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
			str[row] += fmt.Sprintf(" "+f, data[cols*row+col])
		}
		str[row] += "  " + end
	}
	return strings.Join(str, "\n")
}

// common sanity checks
func checkEqualSize(caller string, a, b, out Matrix) {
	rows, cols := a.Rows(), a.Cols()
	if cols != b.Cols() || rows != b.Rows() {
		panic(caller + " - mismatch in no. of rows and columns in input matrices")
	}
	out.Reshape(rows, cols)
}
