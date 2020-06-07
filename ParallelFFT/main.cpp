#include<stdio.h>
#include<time.h>
#include<Windows.h>
#include"FFT.h"
int main(int argc, char** argv)
{
	FFT a = FFT();
	for (int j = 1; j < 30; ++j) {
		ULL NUM = pow(2.0, double(j));
		auto out = new Complex[NUM];
		
		auto in = new double[NUM];
		
		Complex tmp;
		for (int i = 0; i < NUM; ++i) {
			in[i] = double(rand());
		}
		clock_t begin, end;

		begin = clock();
		a.fft_S(in, NUM, out);
		end = clock();
		double time1 = ((double)end - begin) / (double)CLOCKS_PER_SEC;
		
		begin = clock();
		a.fft_P(in, NUM, out);
		end = clock();
		double time2 = ((double)end - begin) / (double)CLOCKS_PER_SEC;

		printf("N=2^%d时,串行花费时间：%.3lfs,并行花费时间：%.3lfs,加速比:%.3lf\n", 
			j, time1, time2, time1 / time2);
		delete []in;
		delete []out;
	}
}