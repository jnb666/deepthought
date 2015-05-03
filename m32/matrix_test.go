package m32

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 1.1, 2.2, 3.3)
	t.Logf("\n%s\n", m)
}

func TestTranspose(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 0, 1, 2, 3, 4, 5)
	m.Transpose()
	t.Logf("\n%s\n", m)
	expect := New(3, 2).Load(RowMajor, 0, 3, 1, 4, 2, 5)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestJoin(t *testing.T) {
	a := New(3, 2).Load(RowMajor, 1)
	b := New(3, 2).Load(RowMajor, 2)
	m := New(3, 4).Join(a, b)
	t.Logf("\n%s\n", m)
	expect := New(3, 4).Load(RowMajor, 1, 1, 2, 2)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestCopy(t *testing.T) {
	m := New(4, 4).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
	t.Logf("\n%s\n", m)
	a := New(4, 2).Copy(m, 1, 3)
	t.Logf("\n%s\n", a)
	b := New(2, 2).CopyRows(a, 1, 3)
	t.Logf("\n%s\n", b)
	expect := New(2, 2).Load(RowMajor, 6, 7, 10, 11)
	if !reflect.DeepEqual(b, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestApply(t *testing.T) {
	m := New(2, 3).Load(RowMajor, 1.1, 2.2, 3.3)
	m.Apply(m, func(x float32) float32 { return 2 * x })
	t.Logf("\n%s\n", m)
	expect := New(2, 3).Load(RowMajor, 2.2, 4.4, 6.6)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestMul(t *testing.T) {
	a := New(2, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6)
	t.Logf("a\n%s\n", a)
	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	t.Logf("b\n%s\n", b)
	m := New(2, 2).Mul(a, b)
	t.Logf("m\n%s\n", m)
	expect := New(2, 2).Load(RowMajor, 58, 64, 139, 154)
	if !reflect.DeepEqual(m, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestMaxCol(t *testing.T) {
	m := New(5, 3).Load(RowMajor, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1)
	t.Logf("\n%s\n", m)
	c := New(5, 1).MaxCol(m)
	t.Logf("\n%s\n", c)
	expect := New(5, 1).Load(RowMajor, 0, 1, 2, 0, 2)
	if !reflect.DeepEqual(c, expect) {
		t.Errorf("expected\n%s\n", expect)
	}
}

func TestSum(t *testing.T) {
	m1 := New(2, 2).Load(RowMajor, 1, -2, 2, -4)
	m2 := New(2, 2).Load(RowMajor, 0, -1, 2, -6)
	m1.Add(-1, m2, m1)
	m1.Apply(m1, func(x float32) float32 { return x * x })
	t.Logf("\n%s\n", m1)
	sum := m1.Sum()
	t.Log(sum)
	if sum != 6 {
		t.Error("wrong sum - expected 6")
	}
}
