package blas

import (
	"reflect"
	"testing"
)

func init() {
	//Init(Native32)
	Init(Native64)
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
}

func TestJoin(t *testing.T) {
	m1 := New(3, 3).Reshape(3, 2).Load(ColMajor, 1, 2, 3)
	m1.SetFormat("%3.0f")
	t.Logf("\n%s\n", m1)
	m2 := New(3, 1).Load(ColMajor, 9)
	m1.Join(m1, m2)
	t.Logf("\n%s\n", m1)
	expect := []float64{1, 1, 9, 2, 2, 9, 3, 3, 9}
	if !reflect.DeepEqual(m1.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
}

func TestSlice(t *testing.T) {
	m1 := New(3, 3).Load(ColMajor, 1, 2, 3)
	m1.SetFormat("%3.0f")
	t.Logf("\n%s\n", m1)
	m1.Slice(1, 2).Load(ColMajor, 6)
	t.Logf("\n%s\n", m1)
	expect := []float64{1, 6, 1, 2, 6, 2, 3, 6, 3}
	if !reflect.DeepEqual(m1.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
}

func TestAdd(t *testing.T) {
	m1 := New(3, 3).Load(RowMajor, 1, 2)
	m2 := New(3, 3).Load(RowMajor, 2, 3, 4)
	m1.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m1)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	m := New(3, 3).Add(m1, m2).Scale(10)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	m.Sub(m, m2.Scale(10)).Scale(0.1)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), m1.Data(RowMajor)) {
		t.Error("expected m==m1")
	}
}

func TestMul(t *testing.T) {
	a := New(2, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6)
	a.SetFormat("%3.0f")
	t.Logf("a\n%s\n", a)
	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	b.SetFormat("%3.0f")
	t.Logf("b\n%s\n", b)
	m := New(2, 2).Mul(a, b, false, false, false)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	expect := []float64{58, 64, 139, 154}
	if m.Rows() != 2 || m.Cols() != 2 || !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("norm: expected", expect)
	}
	// check if a is transposed
	a.Load(ColMajor, 1, 2, 3, 4, 5, 6).Reshape(3, 2)
	t.Logf("a\n%s\n", a)
	m.Mul(a, b, true, false, false)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("atrans: expected", expect)
	}
	// check if b is transposed
	b.Load(ColMajor, 7, 8, 9, 10, 11, 12).Reshape(2, 3)
	t.Logf("b\n%s\n", b)
	m.Mul(a, b, true, true, false)
	t.Logf("m\n%s\n", m)
	if !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("abtrans: expected", expect)
	}
	// check out transposed
	a1 := New(1, 3).Load(RowMajor, 1, 2, 3)
	a1.SetFormat("%3.0f")
	t.Logf("a\n%s\n", a1)
	t.Logf("b\n%s\n", b)
	m.Mul(a1, b, false, true, true)
	t.Logf("m\n%s\n", m)
	if m.Rows() != 2 || m.Cols() != 1 || !reflect.DeepEqual(m.Data(ColMajor), expect[:2]) {
		t.Error("outtrans: expected", expect[:2])
	}
}

func TestApply(t *testing.T) {
	m := New(3, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	fu := Unary64(func(x float64) float64 { return x * x })
	fb := Binary64(func(x, y float64) float64 { return x + y*y })
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
		t.Errorf("expected\n%s\n", expect)
	}
}
