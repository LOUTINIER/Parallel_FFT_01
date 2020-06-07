#pragma once
typedef unsigned long long ULL;

struct Complex
{
	double rl = 0.0;
	double im = 0.0;
	const Complex operator+(const Complex& rhs) const {
		Complex tmp;
		tmp.rl = rl + rhs.rl;
		tmp.im = im + rhs.im;
		return tmp;
	}
	const Complex operator*(const Complex& rhs) const {
		Complex tmp;
		tmp.rl = rl * rhs.rl - im * rhs.im;
		tmp.im = im * rhs.rl + rl * rhs.im;
		return tmp;
	}
};
class FFT {
private:
	const bool inline IsPowerOfTwo(const ULL& numIn) const;
	const int GetExpOfW(const size_t& raw, const size_t& column, const size_t& len) const;
	static constexpr double pi = 3.141592653589793238;
public:
	const bool fft_P(const double* in,const size_t len,Complex* out) const;
	const bool fft_S(const double* in,const size_t len,Complex* out) const;
};