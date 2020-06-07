#include "FFT.h"
#include<stdlib.h>
#include<omp.h>
#include <cstdint>
#include <corecrt_math.h>
inline const bool FFT::IsPowerOfTwo(const ULL& numIn) const
{
	return (numIn & (numIn - 1) ? true : false);
}

const int FFT::GetExpOfW(const size_t& raw, const size_t& column, const size_t& len) const
{
	int tmp = column;
	int out = 0;
	for (size_t i = 0; i < len-1; ++i) {
		out <<= 1;
		out += tmp & 1;
		tmp >>= 1;
	}
	int andObj = 0;
	for (size_t i = 0; i < raw; ++i) {
		andObj <<= 1;
		andObj += 1;
	}
	return (out & andObj) << (len - 1 - raw);
}


const bool FFT::fft_P(const double* in,const size_t len, Complex* out) const
{
	if (!len > 0 || IsPowerOfTwo(len))return false;

	size_t raw = 0;
	for (ULL tmp = len; tmp; tmp /= 2) { ++raw; }

	Complex W_N = Complex();
	W_N.rl = cos(1.0 * 2 / len * pi);
	W_N.im = sin(1.0 * 2 / len * pi);

	Complex unitComplex;
	unitComplex.rl = 1.0;
	unitComplex.im = 0.0;

	Complex* tda = new Complex[len];
	Complex* tdb=nullptr;
#pragma omp parallel num_threads(size_t(raw/1.5))
	for (size_t i = omp_get_thread_num(); i < len; i+=omp_get_num_threads())
	{
		tda[i].rl = in[i];
		tda[i].im = 0.0;
	}

	// Use this to inc little(about 20% to 33%) storage ,but dec much(multiple) CPU
	auto expOfW = new Complex[len];
	expOfW[0] = unitComplex; expOfW[1] = W_N;
	for (size_t i = 2; i < len; ++i) {
		expOfW[i] = (W_N * expOfW[i - 1]);
	}

	for (size_t i = 1; i < raw; ++i) {
		tdb = new Complex[len];
		size_t blockNum = (int64_t)1 << (i - 1);
		size_t half = len / blockNum / 2;
#pragma omp parallel num_threads(32)
		{
			for (size_t z = omp_get_thread_num(); z < blockNum; z += omp_get_num_threads()) {
				size_t itBegin = z * half * 2;
				for (int j = itBegin; j < itBegin + half; ++j) {
					// Use this to dec little storage, inc much CPU
					//size_t expi = GetExpOfW(i, j, raw);
					//size_t expj = GetExpOfW(i, half + j, raw);
					//Complex Ei = unitComplex, Ej = unitComplex;
					//while (expi--) {
					//	Ei = Ei * W_N;
					//}
					//while (expj--) {
					//	Ej = Ej * W_N;
					//}
					tdb[j] = tda[j] +
						expOfW[GetExpOfW(i, j, raw)] * tda[half + j];
					tdb[half + j] = tda[j] +
						expOfW[GetExpOfW(i, half + j, raw)] * tda[half + j];
				}
			}
		}
		delete[]tda;
		tda = tdb;
	}

	delete[]expOfW;

	for (size_t i = 0; i < len; ++i) {
		out[i] = (tdb[GetExpOfW(raw - 1, i, raw)]);
	}
	// 只要释放一次，td[0]和td[1]指向同一个地方
	delete[]tdb;
	return true;
}


const bool FFT::fft_S(const double* in,const size_t len,Complex* out) const
{
	if (!len > 0 || IsPowerOfTwo(len))return false;

	size_t raw = 0;
	for (ULL tmp = len; tmp; tmp /= 2) { ++raw; }

	Complex W_N = Complex();
	W_N.rl = cos(1.0 * 2 / len * pi);
	W_N.im = sin(1.0 * 2 / len * pi);

	Complex unitComplex;
	unitComplex.rl = 1.0;
	unitComplex.im = 0.0;

	Complex* tda = new Complex[len];
	Complex* tdb=nullptr;
	for (size_t i = 0; i < len; ++i)
	{
		tda[i].rl = in[i];
		tda[i].im = 0.0;
	}

	// Use this to delc little(about 20% to 33%) storage ,but increase much(multiple) CPU
	auto expOfW = new Complex[len];
	expOfW[0] = unitComplex; expOfW[1] = W_N;
	for (size_t i = 2; i < len; ++i) {
		expOfW[i] = (W_N * expOfW[i - 1]);
	}

	for (size_t i = 1; i < raw; ++i) {
		tdb = new Complex[len];
		size_t blockNum = (int64_t)1 << (i - 1);
		size_t half = len / blockNum / 2;
		size_t pProportion = 8 * blockNum / (half + blockNum);
		pProportion = pProportion > 0 ? pProportion : 1;
		for (size_t z = 0; z < blockNum; ++z) {
			size_t itBegin = z * half * 2;
			for (int j = itBegin; j < itBegin + half; ++j) {
				/*size_t expi = GetExpOfW(i, j, raw);
				size_t expj = GetExpOfW(i, half + j, raw);
				Complex Ei = unitComplex, Ej = unitComplex;
				while (expi--) {
					Ei = Ei * W_N;
				}
				while (expj--){
					Ej = Ej * W_N;
				}*/
				tdb[j] = tda[j] + expOfW[GetExpOfW(i, j, raw)] * tda[half + j];
				tdb[half + j] = tda[j] + expOfW[GetExpOfW(i, half+j, raw)] * tda[half + j];
			}
		}
		delete []tda;
		tda = tdb;
	}

	delete []expOfW;

	for (size_t i = 0; i < len; ++i) {
		out[i] = (tdb[GetExpOfW(raw - 1, i, raw)]);
	}

	delete []tdb;
	
	return true;
}