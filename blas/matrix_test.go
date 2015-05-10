package blas

import (
	"math"
	"reflect"
	"testing"
)

func init() {
	//Init(Native32)
	//Init(Native64)
	Init(OpenCL32)
}

func TestLoad(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 1, 2, 3)
	m.SetFormat("%3.0f")
	t.Logf("\n%s\n", m)
	expect := []float64{1, 2, 3, 1, 2, 3}
	data := m.Data(RowMajor)
	if !reflect.DeepEqual(data, expect) {
		t.Error("expected", expect, "got", data)
	}
	m.Release()
}

func TestCopy(t *testing.T) {
	m := New(2, 2).Load(RowMajor, 1, 2, 3, 4)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	m2 := New(2, 3).Copy(m)
	m2.Reshape(2, 3, false)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	expect := []float64{1, 2, 0, 3, 4, 0}
	if !reflect.DeepEqual(m2.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
	m2.Release()
	m.Release()
}

func TestAdd(t *testing.T) {
	m1 := New(3, 3).Load(RowMajor, 1, 2)
	m2 := New(3, 3).Load(RowMajor, 2, 3, 4)
	m1.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m1)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	m := New(3, 3).Add(m1, m2, 1).Scale(10)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	m.Add(m, m2.Scale(10), -1).Scale(0.1)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), m1.Data(RowMajor)) {
		t.Error("expected m==m1")
	}
	m1.Release()
	m2.Release()
	m.Release()
}

func TestMElem(t *testing.T) {
	m1 := New(3, 2).Load(ColMajor, 1, 2)
	m2 := New(2, 3).Load(RowMajor, 2, 3, 4)
	m1.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m1)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	m := New(2, 3).MulElem(m1, m2, true, false)
	m.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m)
	m2.Load(RowMajor, 2, 6, 4, 4, 3, 8)
	m1.Cmp(m2, m, 1e-8)
	if m1.Sum() != 0 {
		t.Errorf("expected\n%s", m2)
	}
	m1.Release()
	m2.Release()
	m.Release()
}

func TestSlice(t *testing.T) {
	m := New(3, 3).Load(ColMajor, 1, 2, 3)
	m.SetFormat("%3.0f")
	t.Logf("\n%s\n", m)
	m.Col(1, 2).Load(ColMajor, 4, 0, 6)
	t.Logf("\n%s\n", m)
	m.Row(1, 2).Load(ColMajor, 2, 5, 0)
	t.Logf("\n%s\n", m)
	expect := []float64{1, 2, 3, 4, 5, 6, 1, 0, 3}
	if !reflect.DeepEqual(m.Data(ColMajor), expect) {
		t.Error("expected", expect)
	}
	a := m.Col(0, 2)
	t.Logf("a\n%s\n", a)
	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	b.SetFormat("%3.0f")
	t.Logf("b\n%s\n", b)
	out := New(2, 2).Mul(a, b, true, false)
	out.SetFormat("%3.0f")
	t.Logf("out\n%s\n", out)
	expect = []float64{58, 64, 139, 154}
	if !reflect.DeepEqual(out.Data(RowMajor), expect) {
		t.Error("atrans: expected", expect)
	}
	m.Release()
	b.Release()
	out.Release()
}

func TestMul(t *testing.T) {
	a := New(2, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6)
	a.SetFormat("%3.0f")
	t.Logf("a\n%s\n", a)
	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	b.SetFormat("%3.0f")
	t.Logf("b\n%s\n", b)
	m := New(2, 2).Mul(a, b, false, false)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	expect := []float64{58, 64, 139, 154}
	if m.Rows() != 2 || m.Cols() != 2 || !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("norm: expected", expect)
	}
	// check if a is transposed
	a = New(3, 2).Load(ColMajor, 1, 2, 3, 4, 5, 6)
	a.SetFormat("%3.0f")
	t.Logf("a\n%s\n", a)
	m.Mul(a, b, true, false)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("atrans: expected", expect)
	}
	// check if b is transposed
	b = New(2, 3).Load(ColMajor, 7, 8, 9, 10, 11, 12)
	b.SetFormat("%3.0f")
	t.Logf("b\n%s\n", b)
	m.Mul(a, b, true, true)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("abtrans: expected", expect)
	}
	a.Release()
	b.Release()
	m.Release()
	return
}

func TestApply(t *testing.T) {
	m := New(3, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	var fu UnaryFunction
	var fb BinaryFunction
	if implementation == OpenCL32 {
		fu = NewUnaryCL("x * x")
		fb = NewBinaryCL("x + y*y")
	} else {
		fu = Unary64(func(x float64) float64 { return x * x })
		fb = Binary64(func(x, y float64) float64 { return x + y*y })
	}
	fu.Apply(m, m)
	t.Logf("m\n%s\n", m)
	expect := []float64{1, 4, 9, 16, 25, 36, 49, 64, 81}
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
	m2 := New(3, 3).Load(RowMajor, 1, 2, 3)
	m2.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m2)
	fb.Apply(m, m2, m)
	t.Logf("m\n%s\n", m)
	expect = []float64{2, 8, 18, 17, 29, 45, 50, 68, 90}
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
	m.Release()
	m2.Release()
}

func TestSum(t *testing.T) {
	m := New(5, 3).Load(RowMajor, 1, 2, 3, 4)
	m.SetFormat("%3.0f")
	t.Logf("\n%s\n", m)
	sum := m.Sum()
	t.Log("sum = ", sum)
	if sum != 36 {
		t.Error("wrong sum!")
	}
	m.Release()
}

func TestMaxCol(t *testing.T) {
	m := New(5, 3).Load(RowMajor, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1)
	m.SetFormat("%3.0f")
	t.Logf("\n%s\n", m)
	c := New(5, 1).MaxCol(m)
	c.SetFormat("%3.0f")
	t.Logf("\n%s\n", c)
	expect := []float64{0, 1, 2, 0, 2}
	if !reflect.DeepEqual(c.Data(ColMajor), expect) {
		t.Error("expected", expect)
	}
	m.Release()
	c.Release()
}

func TestNorm(t *testing.T) {
	m := New(3, 3).Load(RowMajor, 2, 1, 1, 3, 0, 0, 5, 2.5, -2.5)
	m.SetFormat("%5.2f")
	t.Logf("\n%s\n", m)
	m.Norm(m)
	t.Logf("\n%s\n", m)
	expect := []float64{0.5, 0.25, 0.25, 1, 0, 0, 1, 0.5, -0.5}
	for i, val := range m.Data(RowMajor) {
		if math.Abs(val-expect[i]) > 1e-5 {
			t.Error("expected", expect[i], "got", val)
		}
	}
	m.Release()
}
