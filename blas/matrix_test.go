package blas

import (
	"math"
	"math/rand"
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
	m := New(4, 2).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	m2 := New(4, 3).Copy(m, nil)
	m2.Reshape(4, 3, false)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	expect := []float64{1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0}
	if !reflect.DeepEqual(m2.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
	// select top and bottom row
	ix := New(2, 1).Load(ColMajor, 0, 3)
	m2.Copy(m, ix)
	t.Logf("m2\n%s\n", m2)
	expect = []float64{1, 2, 7, 8}
	if !reflect.DeepEqual(m2.Data(RowMajor), expect) {
		t.Error("expected", expect)
	}
	ix.Release()
	m2.Release()
	m.Release()
}

func TestTranspose(t *testing.T) {
	expect := []float64{1, 2, 3, 4, 5, 6}
	m := New(2, 3).Load(RowMajor, expect...)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	m1 := New(3, 2).Transpose(m)
	m1.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m1)
	if m1.Rows() != 3 || m1.Cols() != 2 || !reflect.DeepEqual(m1.Data(ColMajor), expect) {
		t.Error("expected", expect)
	}
	m1.Release()
	m.Release()
	big := New(20, 25).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	big.SetFormat("%3.0f")
	t.Logf("big\n%s\n", big)
	tbig := New(20, 25).Transpose(big)
	tbig.SetFormat("%3.0f")
	t.Logf("big\n%s\n", tbig)
	big.Release()
	tbig.Release()
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
	m1 := New(2, 3).Load(RowMajor, 1, 2)
	m2 := New(2, 3).Load(RowMajor, 2, 3, 4)
	m1.SetFormat("%3.0f")
	t.Logf("m1\n%s\n", m1)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	m := New(2, 3).MulElem(m1, m2)
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
	m.Release()
}

func checkMul(t *testing.T, m Matrix, title string, oTrans bool) {
	t.Logf("%s\n%s\n", title, m)
	expect := []float64{58, 64, 139, 154}
	order := RowMajor
	if oTrans {
		order = ColMajor
	}
	if m.Rows() != 2 || m.Cols() != 2 || !reflect.DeepEqual(m.Data(order), expect) {
		t.Error("expected", expect)
	}
}

func TestMul(t *testing.T) {
	a := New(2, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6)
	at := New(3, 2).Transpose(a)
	a.SetFormat("%3.0f")
	t.Logf("a\n%s\n", a)

	b := New(3, 2).Load(RowMajor, 7, 8, 9, 10, 11, 12)
	bt := New(2, 3).Transpose(b)
	b.SetFormat("%3.0f")
	t.Logf("b\n%s\n", b)

	m := New(2, 2)
	m.SetFormat("%3.0f")

	for _, outTrans := range []bool{false, true} {
		m.Mul(a, b, false, false, outTrans)
		checkMul(t, m, "m", outTrans)

		// test with a transposed
		m.Mul(at, b, true, false, outTrans)
		checkMul(t, m, "m - atrans", outTrans)

		// test with b transposed
		m.Mul(a, bt, false, true, outTrans)
		checkMul(t, m, "m - btrans", outTrans)

		// test with a and b transposed
		m.Mul(at, bt, true, true, outTrans)
		checkMul(t, m, "m - atrans + btrans", outTrans)
	}

	a.Release()
	b.Release()
	at.Release()
	bt.Release()
	m.Release()
	return
}

func checkBig(t *testing.T, m Matrix, title string) {
	t.Logf("%s\n%s\n", title, m)
	expect := []float64{
		2453, 2408, 2687, 2708, 2232, 2565, 2537, 2553, 2954, 2805, 2864, 1523, 2778, 2013, 2629, 2654, 2409, 2636, 2539, 2998,
		2772, 2542, 3006, 2682, 2403, 2819, 3008, 2802, 3143, 3150, 3240, 1876, 2857, 2445, 2798, 3099, 2785, 2960, 2824, 3374,
		2590, 2138, 2576, 2294, 1980, 2344, 2404, 2368, 2742, 2649, 2530, 1558, 2738, 2393, 2180, 2363, 2348, 2726, 2596, 3046,
		3088, 2453, 3217, 2443, 2793, 2875, 3236, 2436, 3241, 2942, 2966, 2254, 3211, 2578, 3005, 2681, 2868, 2973, 2997, 3449,
		2615, 2550, 2763, 2747, 2742, 2532, 2482, 2652, 3015, 2881, 2428, 1589, 3053, 2032, 2450, 2823, 2269, 3022, 2857, 3294,
		2291, 1924, 2048, 1869, 2258, 2138, 2478, 1895, 2435, 2214, 2000, 1910, 2344, 1721, 2310, 2057, 1793, 2548, 1999, 2596,
		2469, 2181, 2466, 2381, 2073, 2730, 2172, 1994, 2277, 2311, 2390, 1870, 2394, 1972, 2318, 2183, 1952, 2740, 2219, 2709,
		2621, 2415, 2958, 2677, 2695, 3023, 2947, 2704, 3106, 2885, 2888, 2275, 2810, 2406, 2375, 2746, 2419, 2943, 2943, 3708,
		2702, 2345, 2240, 2404, 2615, 2804, 2365, 2471, 2597, 2230, 2189, 1680, 2646, 2098, 2043, 2451, 1939, 2986, 2256, 3022,
		2994, 2645, 3066, 2928, 2492, 2844, 2712, 2567, 2851, 2968, 2956, 2306, 2442, 2527, 2699, 2760, 2436, 2829, 2794, 3251,
		3310, 2775, 3028, 2883, 2931, 3185, 2994, 2697, 3266, 2981, 2843, 2413, 2905, 2707, 2867, 2792, 2620, 3314, 2696, 3577,
		2917, 2335, 2603, 2239, 2543, 2479, 2652, 2643, 2718, 2553, 2487, 2083, 2762, 2113, 2289, 2638, 2206, 2690, 2864, 3263,
		3065, 2234, 3061, 2677, 2545, 2779, 2686, 2357, 2737, 2793, 2860, 2500, 2796, 2539, 2331, 2414, 2254, 2901, 2668, 3255,
		2876, 2546, 2719, 2290, 2814, 2561, 2571, 2356, 2364, 2696, 2580, 1795, 2574, 2073, 2503, 2567, 2197, 2411, 2275, 2905,
		2926, 2409, 3029, 2507, 2960, 2899, 3016, 2607, 3123, 2896, 2496, 2280, 3160, 2268, 2696, 2650, 2231, 2901, 2580, 3544,
		2613, 2355, 2636, 2116, 2491, 2659, 2583, 2164, 2896, 2673, 2151, 1960, 2689, 1786, 2408, 2360, 2440, 2852, 2685, 2984,
		3203, 2069, 2738, 2123, 2783, 2477, 2719, 2449, 2877, 2623, 2351, 1536, 2993, 2158, 2356, 2133, 2580, 2589, 2582, 3131,
		2944, 2546, 2836, 2696, 2757, 2854, 2824, 2507, 3306, 3107, 3085, 1901, 3095, 2229, 2807, 2810, 2495, 3008, 2725, 3431,
		2590, 2491, 2593, 2572, 2339, 2388, 2309, 1960, 2469, 2532, 2601, 1911, 2520, 2199, 2102, 2434, 2109, 2147, 2466, 2907,
		2717, 2512, 2845, 2563, 2457, 2541, 2715, 2868, 3229, 3025, 2642, 2375, 3201, 2557, 2797, 2801, 2812, 2755, 2885, 3614,
		2932, 2553, 3228, 2611, 2372, 2677, 2561, 2751, 2992, 3392, 3150, 2102, 3061, 2519, 2349, 2907, 2316, 2951, 2759, 3623,
		2892, 2256, 2794, 2533, 2943, 2511, 2508, 2457, 2722, 2672, 2199, 2097, 2803, 2050, 2387, 2428, 2485, 3346, 2890, 3306,
		2660, 2132, 2432, 2319, 2149, 2425, 2149, 2182, 2529, 2542, 2418, 1879, 2376, 2235, 2234, 2463, 1946, 2243, 1934, 2924,
		3441, 2523, 3252, 2649, 2546, 3027, 2910, 2712, 3627, 3340, 3063, 2239, 3267, 2570, 2904, 2397, 2910, 3354, 3011, 3575,
		2858, 2279, 3008, 2469, 2356, 2663, 2976, 2753, 3016, 2877, 2748, 2238, 3447, 2656, 2724, 2824, 2715, 2898, 2835, 3492,
		2845, 2477, 2580, 2861, 2462, 2923, 2525, 2624, 2814, 2807, 2918, 2039, 2704, 2335, 2527, 2792, 2449, 2981, 2806, 3032,
		2677, 2139, 2433, 2363, 2110, 2062, 2228, 2061, 2680, 2674, 2525, 1399, 2227, 1824, 2214, 2424, 2270, 2588, 2505, 2627,
		2629, 2253, 2603, 2177, 2481, 2484, 2234, 2568, 2677, 2512, 2617, 1799, 2628, 1747, 2472, 2593, 2378, 2866, 2552, 3041,
		2634, 2063, 2844, 2662, 2453, 2487, 2718, 2440, 2879, 2910, 2752, 1984, 2640, 2552, 2060, 2582, 2390, 2841, 2749, 3470,
		2682, 2030, 2890, 2194, 2659, 2516, 2714, 2440, 2839, 2802, 2256, 1887, 2710, 2285, 2385, 2304, 2619, 3157, 2576, 3284,
		3274, 3065, 3498, 3133, 2761, 3172, 3080, 2758, 3312, 3464, 3314, 2257, 3162, 2802, 2975, 3335, 2748, 3137, 3111, 3750,
		3256, 3072, 3242, 3159, 2984, 2983, 2541, 2945, 3312, 3480, 3207, 2191, 2915, 2251, 3065, 3274, 2820, 3055, 3063, 3593,
		2797, 2382, 2570, 2546, 2587, 2399, 2741, 2512, 2997, 2577, 2574, 1791, 3020, 2489, 2372, 2572, 2305, 2614, 2658, 3340,
		2515, 1966, 2824, 2030, 2061, 2606, 2319, 2377, 2539, 2565, 2704, 1998, 2509, 1846, 2413, 2258, 2270, 2475, 2479, 2912,
		2838, 2277, 2732, 2534, 2147, 2575, 2576, 2206, 3026, 2886, 2654, 1905, 2412, 2035, 2385, 2454, 2167, 2867, 2833, 2971,
		2261, 1911, 2562, 1895, 1859, 1891, 2096, 1800, 2130, 2430, 2187, 1622, 2195, 1750, 2187, 2208, 1815, 1902, 2240, 2467,
		3238, 2469, 3193, 2530, 2898, 2727, 2846, 2793, 3322, 3347, 2637, 2387, 3133, 2484, 2940, 2776, 2903, 3113, 3074, 3640,
		2607, 1970, 2332, 1764, 2077, 2075, 2120, 1891, 2219, 2472, 2218, 1679, 2480, 1910, 2012, 2079, 1818, 2354, 2163, 2670,
		2939, 2274, 3233, 2475, 2273, 2636, 2762, 2698, 3171, 3384, 2757, 2169, 3229, 2546, 2752, 2685, 2596, 2880, 2910, 3440,
		2851, 2138, 2589, 2212, 2478, 2220, 2592, 2647, 2794, 3021, 2579, 1748, 2444, 2242, 2197, 2589, 1987, 2660, 2543, 3278,
	}
	if m.Rows() != 40 || m.Cols() != 20 || !reflect.DeepEqual(m.Data(RowMajor), expect) {
		t.Error("mul: expected", expect)
	}
}

func TestBigMlt(t *testing.T) {
	rand.Seed(1)
	a := New(40, 30).Load(RowMajor, randSlice(40*30)...)
	at := New(30, 40).Transpose(a)
	a.SetFormat("%2.0f")
	t.Logf("a\n%s\n", a)
	b := New(30, 20).Load(RowMajor, randSlice(30*20)...)
	bt := New(20, 30).Transpose(b)
	b.SetFormat("%2.0f")
	t.Logf("m2\n%s\n", b)
	m := New(40, 20)
	m.SetFormat("%4.0f")

	m.Mul(a, b, false, false, false)
	checkBig(t, m, "m")

	// test with a transposed
	m.Mul(at, b, true, false, false)
	checkBig(t, m, "m - atrans")

	// test with b transposed
	m.Mul(a, bt, false, true, false)
	checkBig(t, m, "m - btrans")

	// test with a and b transposed
	m.Mul(at, bt, true, true, false)
	checkBig(t, m, "m - atrans + btrans")

	a.Release()
	at.Release()
	b.Release()
	bt.Release()
	m.Release()
}

func TestApply(t *testing.T) {
	m := New(3, 3).Load(RowMajor, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	var fu UnaryFunction
	var fb BinaryFunction
	if implementation == OpenCL32 {
		fu = NewUnaryCL("float y = x * x;")
		fb = NewBinaryCL("float z = x + y*y;")
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
	t.Logf("m\n%s\n", m)
	sum := m.Sum()
	t.Log("sum = ", sum)
	if sum != 36 {
		t.Error("wrong sum!")
	}
	m2 := New(20, 20).Load(RowMajor, 1, 2, 3)
	m2.SetFormat("%3.0f")
	t.Logf("m2\n%s\n", m2)
	sum = m2.Sum()
	t.Log("sum = ", sum)
	if sum != 799 {
		t.Error("wrong sum!")
	}
	m2.SumRows(m)
	t.Logf("m2\n%s\n", m2)
	expect := []float64{6, 7, 8, 9, 6}
	if m2.Rows() != m.Rows() || m2.Cols() != 1 || !reflect.DeepEqual(m2.Data(ColMajor), expect) {
		t.Error("wrong SumRows!")
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

func TestHistogram(t *testing.T) {
	m := New(10, 1).Load(ColMajor, 5, 3, 1, 1, 1, 2, 2, 3, 4, 5)
	m.SetFormat("%3.0f")
	t.Logf("m\n%s\n", m)
	h := New(5, 1).Histogram(m, 5, 0.5, 5.5)
	h.SetFormat("%3.0f")
	t.Logf("hist\n%s\n", h)
	expect := []float64{3, 2, 2, 1, 2}
	if !reflect.DeepEqual(h.Data(ColMajor), expect) {
		t.Error("expected", expect)
	}
}

var m0 Matrix

func randSlice(n int) []float64 {
	res := make([]float64, n)
	for i := range res {
		res[i] = float64(rand.Intn(20))
	}
	return res
}

func BenchmarkAdd(b *testing.B) {
	rows, cols := 1024, 1024
	m0 = New(rows, cols)
	m1 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	m2 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	for i := 0; i < b.N; i++ {
		m0.Add(m1, m2, 1)
		Sync()
	}
}

func BenchmarkTrans(b *testing.B) {
	rows, cols := 1024, 1024
	m0 = New(rows, cols)
	m1 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	for i := 0; i < b.N; i++ {
		m0.Transpose(m1)
		Sync()
	}
}

func BenchmarkMul(b *testing.B) {
	rows, cols := 1024, 1024
	m0 = New(rows, cols)
	m1 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	m2 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	for i := 0; i < b.N; i++ {
		m0.Mul(m1, m2, false, false, false)
		Sync()
	}
}

var total float64

func BenchmarkSum(b *testing.B) {
	rows, cols := 1024, 1024
	m0 := New(rows, cols).Load(RowMajor, randSlice(rows*cols)...)
	for i := 0; i < b.N; i++ {
		total = m0.Sum()
	}
}
