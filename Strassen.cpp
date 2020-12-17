#include<math.h>
#include<stdio.h>
#include<random>
#include<ctime>
#include<chrono>
#include<ratio>

using namespace std;

random_device r;
mt19937 engine(r());
uniform_real_distribution<float> uniform_dist(1, 16);


void Show(float x) {
	printf("%f ", x);
}
int log2(int x) {
	int result = 1;
	while ((x >>= 1) != 0) result++;
	return result;
}
int getNewDimension(int n) {
	return 1 << log2(n);
}
float** additionToLog2Matrix(float** a, int n, int aN) {
	float** result = new float* [n];
	for (int i = 0; i < n; ++i) {
		result[i] = new float[n];
		for (int j = 0; j < n; ++j)
			if (i >= aN || j >= aN)
				result[i][j] = 0;
			else
				result[i][j] = a[i][j];
	}

	return result;
}
void splitMatrix(float** a, float** a11, float** a12, float** a21, float** a22, int aN) {
	int n = aN >> 1;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a11[i][j] = a[i][j];
			a12[i][j] = a[i][j + n];
			a21[i][j] = a[i + n][j];
			a22[i][j] = a[i + n][j + n];
		}
	}
}
float** collectMatrix(float** a11, float** a12, float** a21, float** a22, int n) {
	float** a = new float* [n << 1];
	for (int i = 0; i < (n << 1); i++)
		a[i] = new float[n << 1];

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] = a11[i][j];
			a[i][j + n] = a12[i][j];
			a[i + n][j] = a21[i][j];
			a[i + n][j + n] = a22[i][j];
		}
	}
	delete[] a11, a12, a21, a22;
	return a;
}

float** multiply(float** a, float** b, int N) {
	float** result = new float* [N];
	for (int i = 0; i < N; i++) {
		result[i] = new float[i];
		for (int j = 0; j < N; j++) {
			result[i][j] = 0;
			for (int k = 0; k < N; k++) {
				result[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return result;
}
float** summation(float** a, float** b, int n) {
	float** result = new float* [n];
	for (int i = 0; i < n; i++) {
		result[i] = new float[i];
		for (int j = 0; j < n; j++) {
			result[i][j] = a[i][j] + b[i][j];
		}
	}
	return result;
}
float** subtraction(float** a, float** b, int n) {
	float** result = new float* [n];
	for (int i = 0; i < n; i++) {
		result[i] = new float[i];
		for (int j = 0; j < n; j++) {
			result[i][j] = a[i][j] - b[i][j];
		}
	}
	return result;
}
float** multiStrassen(float** a, float** b, int N) {
	if (N <= 64) {
		return multiply(a, b, N);
	}

	int n = N >> 1;

	float** a11 = new float* [n];
	float** a12 = new float* [n];
	float** a21 = new float* [n];
	float** a22 = new float* [n];

	float** b11 = new float* [n];
	float** b12 = new float* [n];
	float** b21 = new float* [n];
	float** b22 = new float* [n];

	for (int j = 0; j < n; j++) {
		a11[j] = new float[n];
		a12[j] = new float[n];
		a21[j] = new float[n];
		a22[j] = new float[n];

		b11[j] = new float[n];
		b12[j] = new float[n];
		b21[j] = new float[n];
		b22[j] = new float[n];
	}

	splitMatrix(a, a11, a12, a21, a22, N);
	splitMatrix(b, b11, b12, b21, b22, N);

	float** p1 = multiStrassen(summation(a11, a22, n), summation(b11, b22, n), n);
	float** p2 = multiStrassen(summation(a21, a22, n), b11, n);
	float** p3 = multiStrassen(a11, subtraction(b12, b22, n), n);
	float** p4 = multiStrassen(a22, subtraction(b21, b11, n), n);
	float** p5 = multiStrassen(summation(a11, a12, n), b22, n);
	float** p6 = multiStrassen(subtraction(a21, a11, n), summation(b11, b12, n), n);
	float** p7 = multiStrassen(subtraction(a12, a22, n), summation(b21, b22, n), n);

	float** c11 = summation(summation(p1, p4, n), subtraction(p7, p5, n), n);
	float** c12 = summation(p3, p5, n);
	float** c21 = summation(p2, p4, n);
	float** c22 = summation(subtraction(p1, p2, n), summation(p3, p6, n), n);
	delete[] p1, p2, p3, p4, p5, p6, p7;
	return collectMatrix(c11, c12, c21, c22, n);
	delete[] c12, c21, c22;
}
void InitMatrices2(float** a, float** b, int aN)
{
	for (int i = 0; i < aN; i++)
		for (int j = 0; j < aN; j++)
		{
			a[i][j] = uniform_dist(engine);
			b[i][j] = uniform_dist(engine);
		}
}

void SolveStrassen() {
	int N = 8;
	float** a = new float* [N];
	float** b = new float* [N];
	for (int i = 0; i < N; i++) {
		a[i] = new float[N];
		b[i] = new float[N];
	}
	InitMatrices2(a, b, N);
	float** c = multiStrassen(a, b, N);
	printf("Srtrassen C\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			Show(c[i][j]);
		printf("\n");
	}
	printf("Check\n");
	float** C2 = multiply(a, b, N);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Show(C2[i][j]);
		}
		printf("\n");
	}
}

int main(int* argv, char** argc) {
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	SolveStrassen();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

	printf("It took me %lld seconds", static_cast<long long int>(time_span.count()));
	system("pause");
}